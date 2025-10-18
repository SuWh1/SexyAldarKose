import { HackathonHero } from '../components/HackathonHero';
import { HackathonFooter } from '../components/HackathonFooter';
import { SolutionOverview } from '../components/SolutionOverview';
import { TeamSection } from '../components/TeamSection';

export function HomePage() {
  return (
    <>
      <main>
        <HackathonHero />
        <SolutionOverview />
        <TeamSection />
      </main>
      
      <HackathonFooter />
    </>
  );
}
