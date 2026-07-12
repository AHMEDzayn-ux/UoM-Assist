import axios from "axios";

// Base URL — env-driven so it works in production (no hardcoded localhost).
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
export const API_BASE = API_BASE_URL;

// WebSocket base (voice call). Derives ws:// or wss:// from the HTTP base.
export const WS_BASE = API_BASE_URL.replace(/^http/, "ws");

// ---- Admin session token ----------------------------------------------------
const TOKEN_KEY = "admin_token";
export const getToken = () => localStorage.getItem(TOKEN_KEY);
export const setToken = (t) => localStorage.setItem(TOKEN_KEY, t);
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);
export const isAuthenticated = () => !!getToken();

// Authenticated axios instance for admin/console calls.
const api = axios.create({ baseURL: API_BASE_URL });
api.interceptors.request.use((config) => {
  const t = getToken();
  if (t) config.headers.Authorization = `Bearer ${t}`;
  return config;
});
api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response && err.response.status === 401) clearToken();
    return Promise.reject(err);
  }
);

// ---- Auth -------------------------------------------------------------------
export const adminLogin = async (email, password) => {
  const res = await axios.post(`${API_BASE_URL}/api/auth/login`, { email, password });
  setToken(res.data.token);
  return res.data;
};
export const adminRegister = async (email, password, name = "") => {
  const res = await axios.post(`${API_BASE_URL}/api/auth/register`, { email, password, name });
  setToken(res.data.token);
  return res.data;
};
export const logout = () => clearToken();
// Who am I? Returns { id, email, name, is_superadmin, role, client_slug }.
export const getMe = async () => (await api.get("/api/auth/me")).data;

// ---- Domains (public metadata) ----------------------------------------------
export const listDomains = async () =>
  (await axios.get(`${API_BASE_URL}/api/domains`)).data;

// ---- Clients (admin) --------------------------------------------------------
export const createClient = async (payload) =>
  (await api.post("/api/clients", payload)).data;
export const listClients = async (skip = 0, limit = 100) =>
  (await api.get("/api/clients", { params: { skip, limit } })).data;
export const getClient = async (slug) =>
  (await api.get(`/api/clients/${slug}`)).data;
export const updateClient = async (slug, payload) =>
  (await api.patch(`/api/clients/${slug}`, payload)).data;
export const deleteClient = async (slug) =>
  (await api.delete(`/api/clients/${slug}`)).data;

// ---- Documents (admin) ------------------------------------------------------
export const uploadDocument = async (slug, file, category = "general", docType = "document") => {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("category", category);
  fd.append("doc_type", docType);
  const res = await api.post(`/api/clients/${slug}/documents`, fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};
export const listDocuments = async (slug) =>
  (await api.get(`/api/clients/${slug}/documents`)).data;
export const clearDocuments = async (slug) =>
  (await api.delete(`/api/clients/${slug}/documents`)).data;

// ---- Escalations (admin inbox) ----------------------------------------------
export const listEscalations = async (slug, statusFilter) => {
  const params = statusFilter ? { status_filter: statusFilter } : {};
  const res = await api.get(`/api/clients/${slug}/escalations`, { params });
  return res.data;
};
export const resolveEscalation = async (slug, escalationId) =>
  (await api.post(`/api/clients/${slug}/escalations/${escalationId}/resolve`)).data;

// ---- Public (customer-facing, scoped to one slug) ---------------------------
export const publicGetConfig = async (slug) =>
  (await axios.get(`${API_BASE_URL}/api/public/${slug}/config`)).data;
export const publicChat = async (slug, message, history = [], sessionId = null) =>
  (
    await axios.post(`${API_BASE_URL}/api/public/${slug}/chat`, {
      message,
      history,
      session_id: sessionId,
      use_retrieval: true,
      top_k: 4,
    })
  ).data;

export const submitFeedback = async (slug, interactionId, rating) =>
  (
    await axios.post(`${API_BASE_URL}/api/public/${slug}/feedback`, {
      interaction_id: interactionId,
      rating,
    })
  ).data;

// ---- Learning loop (admin) --------------------------------------------------
export const getInsights = async (slug) =>
  (await api.get(`/api/clients/${slug}/insights`)).data;
export const getGaps = async (slug) =>
  (await api.get(`/api/clients/${slug}/gaps`)).data;
export const draftGapAnswer = async (slug, questions) =>
  (await api.post(`/api/clients/${slug}/gaps/draft`, { questions })).data;
export const addKbEntry = async (slug, title, content, tags = []) =>
  (await api.post(`/api/clients/${slug}/kb-entry`, { title, content, tags })).data;

// ---- Transactional actions (admin) ------------------------------------------
export const getRequests = async (slug, statusFilter) => {
  const params = statusFilter ? { status_filter: statusFilter } : {};
  return (await api.get(`/api/clients/${slug}/requests`, { params })).data;
};
export const setRequestStatus = async (slug, actionId, status) =>
  (await api.post(`/api/clients/${slug}/requests/${actionId}/status`, { status })).data;
export const getAccounts = async (slug) =>
  (await api.get(`/api/clients/${slug}/accounts`)).data;
export const seedAccounts = async (slug) =>
  (await api.post(`/api/clients/${slug}/accounts/seed`)).data;

// ---- Health -----------------------------------------------------------------
export const checkHealth = async () =>
  (await axios.get(`${API_BASE_URL}/health`)).data;

export default api;
