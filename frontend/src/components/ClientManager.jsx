import { useState, useEffect } from 'react';
import { createClient, listClients, deleteClient, listDomains } from '../services/api';
import Icon from './Icon';
import './ClientManager.css';

const slugify = (s) =>
    s.toLowerCase().trim().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '');

const ClientManager = ({ onClientSelect, selectedClient }) => {
    const [clients, setClients] = useState([]);
    const [domains, setDomains] = useState([]);
    const [form, setForm] = useState({ name: '', slug: '', description: '', domain: 'generic', persona: '' });
    const [slugEdited, setSlugEdited] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        loadClients();
        listDomains().then((d) => setDomains(d.domains || [])).catch(() => {});
    }, []);

    const loadClients = async () => {
        try {
            const data = await listClients(0, 200);
            setClients(data.clients || []);
        } catch (err) {
            setError('Failed to load clients: ' + (err?.response?.data?.detail || err.message));
        }
    };

    const setField = (k, v) => setForm((f) => ({ ...f, [k]: v }));

    // Auto-derive slug from name until the user edits slug directly.
    const onNameChange = (v) => {
        setField('name', v);
        if (!slugEdited) setField('slug', slugify(v));
    };

    const handleCreate = async (e) => {
        e.preventDefault();
        if (!form.slug.trim()) { setError('Slug is required'); return; }
        setLoading(true);
        setError('');
        try {
            await createClient({
                slug: form.slug,
                name: form.name || form.slug,
                description: form.description,
                domain: form.domain,
                persona: form.persona || null,
            });
            setForm({ name: '', slug: '', description: '', domain: 'generic', persona: '' });
            setSlugEdited(false);
            await loadClients();
        } catch (err) {
            setError('Create failed: ' + (err?.response?.data?.detail || err.message));
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (slug) => {
        if (!window.confirm(`Delete client "${slug}" and all its data?`)) return;
        setLoading(true);
        try {
            await deleteClient(slug);
            if (selectedClient === slug) onClientSelect(null);
            await loadClients();
        } catch (err) {
            setError('Delete failed: ' + (err?.response?.data?.detail || err.message));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="client-manager">
            <h2><Icon name="plus" size={18} /> Create a client</h2>
            {error && <div className="error-message">{error}</div>}

            <form onSubmit={handleCreate} className="create-client-form">
                <label>Name
                    <input type="text" placeholder="e.g. Acme Support" value={form.name}
                        onChange={(e) => onNameChange(e.target.value)} disabled={loading} />
                </label>
                <label>Slug (URL id)
                    <input type="text" placeholder="acme-support" value={form.slug}
                        onChange={(e) => { setSlugEdited(true); setField('slug', slugify(e.target.value)); }}
                        disabled={loading} />
                </label>
                <label>Domain
                    <select value={form.domain} onChange={(e) => setField('domain', e.target.value)} disabled={loading}>
                        {domains.map((d) => (
                            <option key={d.key} value={d.key}>{d.display_name}</option>
                        ))}
                    </select>
                </label>
                <label>Description
                    <input type="text" placeholder="Short description" value={form.description}
                        onChange={(e) => setField('description', e.target.value)} disabled={loading} />
                </label>
                <label className="full">Persona override (optional — defaults to the domain's persona)
                    <textarea rows="2" placeholder="Leave blank to use the domain default"
                        value={form.persona} onChange={(e) => setField('persona', e.target.value)} disabled={loading} />
                </label>
                <button type="submit" disabled={loading} className="btn-primary">
                    {loading ? 'Creating…' : 'Create Client'}
                </button>
            </form>

            <div className="clients-list">
                <h3>Your Clients ({clients.length})</h3>
                {clients.length === 0 ? (
                    <p className="empty-state">No clients yet. Create one above to get started!</p>
                ) : (
                    <div className="clients-grid">
                        {clients.map((c) => (
                            <div key={c.slug} className={`client-card ${selectedClient === c.slug ? 'selected' : ''}`}>
                                <div className="client-info">
                                    <h4>{c.name || c.slug}</h4>
                                    <span className={`domain-badge domain-${c.domain}`}>{c.domain}</span>
                                    {c.description && <p>{c.description}</p>}
                                    <small>{c.document_count || 0} document(s) · /c/{c.slug}</small>
                                </div>
                                <div className="client-actions">
                                    <button onClick={() => onClientSelect(c.slug)} className="btn-select"
                                        disabled={selectedClient === c.slug}>
                                        {selectedClient === c.slug ? 'Selected' : 'Manage'}
                                    </button>
                                    <button onClick={() => handleDelete(c.slug)} className="btn-delete" disabled={loading}>
                                        Delete
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default ClientManager;
