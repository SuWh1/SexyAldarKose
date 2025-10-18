import { HackathonHeader } from './components/HackathonHeader';
import { HackathonHero } from './components/HackathonHero';
import { HackathonFooter } from './components/HackathonFooter';
import { SolutionOverview } from './components/SolutionOverview';
import { TeamSection } from './components/TeamSection';

function App() {
  return (
    <div className="min-h-screen bg-white">
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
