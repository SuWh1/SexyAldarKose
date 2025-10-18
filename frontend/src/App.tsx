import { HackathonHeader } from './components/HackathonHeader';
import { HackathonHero } from './components/HackathonHero';
import { HackathonFooter } from './components/HackathonFooter';
import { SolutionOverview } from './components/SolutionOverview';
import { TeamSection } from './components/TeamSection';

function App() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'rgba(0, 0, 0, 1)' }}>
      <HackathonHeader />
      
      <main>
        <HackathonHero />
        <SolutionOverview />
        <TeamSection />
      </main>
      
      <HackathonFooter />
    </div>
  );
}

export default App;
