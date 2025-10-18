import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { HackathonHeader } from './components/HackathonHeader';
import { HomePage } from './pages/HomePage';
import { ComicsPage } from './pages/ComicsPage';
import { GeneratePage } from './pages/GeneratePage';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen" style={{ backgroundColor: 'rgba(0, 0, 0, 1)' }}>
        <HackathonHeader />
        
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/aldarkose" element={<ComicsPage />} />
          <Route path="/generate" element={<GeneratePage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
