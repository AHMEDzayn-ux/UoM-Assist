import { Routes, Route, Navigate } from 'react-router-dom';
import AdminConsole from './pages/AdminConsole';
import CustomerApp from './pages/CustomerApp';
import './App.css';

function App() {
    return (
        <Routes>
            {/* Operator console (password-gated) */}
            <Route path="/admin" element={<AdminConsole />} />
            {/* Customer-facing assistant, scoped to a single client slug */}
            <Route path="/c/:slug" element={<CustomerApp />} />
            {/* Default: operator console */}
            <Route path="/" element={<Navigate to="/admin" replace />} />
            <Route path="*" element={<Navigate to="/admin" replace />} />
        </Routes>
    );
}

export default App;
