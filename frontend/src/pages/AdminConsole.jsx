import { useState } from 'react';
import { isAuthenticated, logout } from '../services/api';
import AdminLogin from '../components/AdminLogin';
import ClientManager from '../components/ClientManager';
import DocumentUpload from '../components/DocumentUpload';
import DeployPanel from '../components/DeployPanel';
import WhatsAppConfig from '../components/WhatsAppConfig';
import ChatInterface from '../components/ChatInterface';
import VoiceCall from '../components/VoiceCall';
import EscalationInbox from '../components/EscalationInbox';
import RequestsInbox from '../components/RequestsInbox';
import InsightsPanel from '../components/InsightsPanel';
import Icon from '../components/Icon';
import './AdminConsole.css';

const TABS = [
    { key: 'knowledge', label: 'Knowledge', icon: 'book' },
    { key: 'insights', label: 'Insights', icon: 'chart' },
    { key: 'inbox', label: 'Inbox', icon: 'inbox' },
    { key: 'requests', label: 'Requests', icon: 'ticket' },
    { key: 'deploy', label: 'Deploy', icon: 'link' },
    { key: 'whatsapp', label: 'WhatsApp', icon: 'message' },
    { key: 'test', label: 'Test chat', icon: 'sparkle' },
];

const AdminConsole = () => {
    const [authed, setAuthed] = useState(isAuthenticated());
    const [selectedClient, setSelectedClient] = useState(null);
    const [tab, setTab] = useState('knowledge');
    const [testVoice, setTestVoice] = useState(false);

    if (!authed) {
        return <AdminLogin onLogin={() => setAuthed(true)} />;
    }

    const handleLogout = () => {
        logout();
        setAuthed(false);
        setSelectedClient(null);
    };

    return (
        <div className="console">
            <header className="console-header">
                <div className="console-title">
                    <span className="console-logo" aria-hidden="true"><Icon name="sparkle" size={20} /></span>
                    <div>
                        <h1>Operator Console</h1>
                        <p>Configure and deploy AI customer-care assistants for your clients.</p>
                    </div>
                </div>
                <button className="btn-logout" onClick={handleLogout}>
                    <Icon name="logout" size={15} />
                    Sign out
                </button>
            </header>

            <main className="console-main">
                <section className="console-left">
                    <ClientManager
                        onClientSelect={(slug) => { setSelectedClient(slug); setTab('knowledge'); }}
                        selectedClient={selectedClient}
                    />
                </section>

                <section className="console-right">
                    {!selectedClient ? (
                        <div className="console-placeholder">
                            <Icon name="arrow-left" size={26} />
                            <h2>Select or create a client</h2>
                            <p>Pick a client to upload its knowledge base, configure WhatsApp, grab its deploy links, and test the assistant.</p>
                        </div>
                    ) : (
                        <>
                            <div className="client-tabs">
                                <div className="client-tabs-title">Managing: <strong>{selectedClient}</strong></div>
                                <div className="tab-row">
                                    {TABS.map((t) => (
                                        <button key={t.key}
                                            className={`tab-btn ${tab === t.key ? 'active' : ''}`}
                                            onClick={() => setTab(t.key)}>
                                            <Icon name={t.icon} size={15} />
                                            {t.label}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="tab-content">
                                {tab === 'knowledge' && <DocumentUpload clientId={selectedClient} />}
                                {tab === 'insights' && <InsightsPanel slug={selectedClient} />}
                                {tab === 'inbox' && <EscalationInbox slug={selectedClient} />}
                                {tab === 'requests' && <RequestsInbox slug={selectedClient} />}
                                {tab === 'deploy' && <DeployPanel slug={selectedClient} />}
                                {tab === 'whatsapp' && <WhatsAppConfig slug={selectedClient} />}
                                {tab === 'test' && (
                                    <div>
                                        <div className="test-mode-row">
                                            <button className={`tab-btn ${!testVoice ? 'active' : ''}`}
                                                onClick={() => setTestVoice(false)}><Icon name="message" size={15} /> Text</button>
                                            <button className={`tab-btn ${testVoice ? 'active' : ''}`}
                                                onClick={() => setTestVoice(true)}><Icon name="mic" size={15} /> Voice</button>
                                        </div>
                                        {testVoice ? (
                                            <VoiceCall key={selectedClient} slug={selectedClient}
                                                onClose={() => setTestVoice(false)} />
                                        ) : (
                                            <ChatInterface clientId={selectedClient} isPublic />
                                        )}
                                    </div>
                                )}
                            </div>
                        </>
                    )}
                </section>
            </main>
        </div>
    );
};

export default AdminConsole;
