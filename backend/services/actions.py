"""
Transactional actions — the agent's "do things" tools, per-domain (mock-first).

An action is just a TOOL we hand the LLM: a name + description + JSON parameter
schema (same shape as `search_knowledge_base` / `escalate_to_human`). The agent
decides when to call one and collects the arguments conversationally. This module:

  - declares the per-domain tool set        -> get_action_tools(domain)
  - executes a called tool by its KIND      -> execute_action(...)

Every tool maps to one of four generic KINDS, so adding a domain/action is data,
not new execution code:

  ticket          -> log a request, return a reference number
  callback        -> log a callback request, return a reference
  account_lookup  -> read a (mock) account, return its details   (read-only, not recorded)
  account_change  -> update a (mock) account, return confirmation (recorded)

MOCK-FIRST: account handlers read/write the seeded `MockAccount` table. Swap a
handler body for a real API call to go live — nothing else changes.
"""

from typing import Any, Dict, List, Optional

from database import SessionLocal
from services import client_store
from logger import get_logger

logger = get_logger(__name__)

# Reference prefixes per kind (e.g. TKT-1042).
_REF_PREFIX = {"ticket": "TKT", "callback": "CB", "account_change": "CHG"}


def _tool(name: str, kind: str, description: str, properties: dict,
          required: List[str], confirm: bool = False) -> dict:
    """Build one action spec (OpenAI function-calling schema + our metadata)."""
    return {
        "kind": kind,
        "confirm": confirm,
        "schema": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        },
    }


# Reusable parameter fragments.
_IDENT_GENERIC = {"identifier": {"type": "string", "description": "The customer's account email or ID"}}


# ---- Per-domain registry ----------------------------------------------------

ACTIONS: Dict[str, List[dict]] = {
    "university": [
        _tool("create_request", "ticket",
              "Log a student request or issue (document request, complaint, general query to "
              "record). Gather a clear subject and details first.",
              {"subject": {"type": "string", "description": "Short title of the request"},
               "details": {"type": "string", "description": "Details of the request"}},
              ["subject"]),
        _tool("book_advisor", "callback",
              "Book a callback / advisor appointment for the student.",
              {"phone": {"type": "string", "description": "Contact number"},
               "topic": {"type": "string", "description": "What they need advice on"},
               "name": {"type": "string", "description": "Student name if given"}},
              ["phone"]),
        _tool("check_request", "request_status",
              "Check the status/progress of a previously logged request, appointment, or change "
              "using its reference number (e.g. TKT-1005, CB-1002).",
              {"reference": {"type": "string", "description": "The reference number to check"}},
              ["reference"]),
    ],
    "generic": [
        _tool("create_ticket", "ticket",
              "Log a support ticket for any request or issue the customer wants recorded.",
              {"subject": {"type": "string", "description": "Short title of the issue"},
               "details": {"type": "string", "description": "What the customer reported"}},
              ["subject"]),
        _tool("request_callback", "callback",
              "Schedule a callback from a human agent.",
              {"phone": {"type": "string", "description": "Number to call back"},
               "topic": {"type": "string", "description": "What it's about"},
               "name": {"type": "string", "description": "Customer name if given"}},
              ["phone"]),
        _tool("lookup_account", "account_lookup",
              "Look up the customer's account to report their status, plan, or balance. "
              "Requires an account email or ID.",
              dict(_IDENT_GENERIC), ["identifier"]),
        _tool("update_account", "account_change",
              "Make a change to the customer's account. Confirm the exact change with the "
              "customer BEFORE calling this. Requires an account email or ID.",
              {**_IDENT_GENERIC, "change": {"type": "string", "description": "The change to make"}},
              ["identifier", "change"], confirm=True),
        _tool("check_ticket", "request_status",
              "Check the status/progress of a previously logged ticket, callback, or change "
              "using its reference number (e.g. TKT-1005).",
              {"reference": {"type": "string", "description": "The reference number to check"}},
              ["reference"]),
    ],
}


def _specs(domain: str) -> List[dict]:
    return ACTIONS.get((domain or "generic").lower(), ACTIONS["generic"])


def _spec(domain: str, name: str) -> Optional[dict]:
    for s in _specs(domain):
        if s["schema"]["function"]["name"] == name:
            return s
    return None


def get_action_tools(domain: str) -> List[dict]:
    """OpenAI function-schema tool defs to bind for this domain's agent."""
    return [s["schema"] for s in _specs(domain)]


def is_action(domain: str, name: str) -> bool:
    return _spec(domain, name) is not None


# ---- Execution --------------------------------------------------------------

def _format_account(acct) -> str:
    data = acct.data or {}
    parts = [f"{k.replace('_', ' ')}: {v}" for k, v in data.items()]
    who = acct.name or acct.identifier
    return f"Account for {who} — " + "; ".join(parts) if parts else f"Account for {who} found."


def execute_action(client_slug: str, session_id: Optional[str], name: str,
                   args: Dict[str, Any], domain: str) -> str:
    """Run a called action and return a short result string for the agent to relay.

    Opens its own DB session (like _record_escalation). Never raises — returns a
    plain message the model can read out even on failure.
    """
    spec = _spec(domain, name)
    if spec is None:
        return "That action isn't available."
    kind = spec["kind"]
    args = args or {}
    db = SessionLocal()
    try:
        if kind in ("ticket", "callback"):
            row = client_store.create_action_request(
                db, client_slug=client_slug, session_id=session_id,
                action_type=name, kind=kind, payload=args,
            )
            row.reference = f"{_REF_PREFIX[kind]}-{1000 + row.id}"
            db.commit()
            what = "callback" if kind == "callback" else "request"
            return (f"Logged the {what} — reference {row.reference}. "
                    "A human agent will follow up.")

        if kind == "request_status":
            ref = (args.get("reference") or "").strip()
            row = client_store.get_action_by_reference(db, client_slug, ref)
            if row is None:
                return (f"No request was found with reference '{ref}'. Ask the customer to "
                        "double-check the number.")
            when = row.created_at.strftime("%b %d, %Y") if row.created_at else "recently"
            state = "completed" if row.status == "done" else "open (being worked on)"
            subj = (row.payload or {}).get("subject") or (row.payload or {}).get("topic") or row.action_type
            return (f"{row.reference} ({subj}) was logged on {when} and is currently {state}.")

        identifier = (args.get("identifier") or "").strip()
        if not identifier:
            return "I need the account identifier (phone/email/ID) first — please ask the customer for it."

        acct = client_store.get_mock_account(db, client_slug, identifier)
        if acct is None:
            return f"No account was found for '{identifier}'. Double-check the details with the customer."

        if kind == "account_lookup":
            return _format_account(acct)   # read-only, not recorded

        # account_change: apply the mock change, record it, confirm.
        data = dict(acct.data or {})
        if args.get("new_plan"):
            data["plan"] = args["new_plan"]
            change_desc = f"plan changed to {args['new_plan']}"
        elif args.get("change"):
            data.setdefault("recent_changes", [])
            data["recent_changes"] = (data.get("recent_changes") or [])[-4:] + [args["change"]]
            change_desc = args["change"]
        else:
            change_desc = "account updated"
        client_store.upsert_mock_account(db, client_slug, acct.identifier, name=acct.name, data=data)
        row = client_store.create_action_request(
            db, client_slug=client_slug, session_id=session_id,
            action_type=name, kind=kind, payload=args, result=change_desc,
        )
        row.reference = f"{_REF_PREFIX['account_change']}-{1000 + row.id}"
        db.commit()
        return f"Done — {change_desc}, confirmed. Reference {row.reference}."
    except Exception as e:
        logger.warning(f"Action '{name}' failed for {client_slug}: {e}")
        return "Sorry, I couldn't complete that action just now."
    finally:
        db.close()
