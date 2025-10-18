import { Highlight } from './Highlight';

export function HackathonFooter() {
  return (
    <footer className="text-white py-12 animate-in fade-in duration-800" style={{ backgroundColor: '#141414' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Brand */}
          <div className="animate-in slide-in-from-bottom duration-800 delay-200">
            <h3 className="text-2xl font-bold mb-4 transition-transform duration-300 hover:scale-105 cursor-pointer">
              Team <span style={{ color: '#C1F21D' }} className="transition-colors duration-300 hover:brightness-110">Richards</span>
            </h3>
            <p className="text-white opacity-80 transition-opacity duration-300 hover:opacity-100">
              Building <Highlight>privacy-first</Highlight> solutions for geotrack analytics at Decentrathon 2025
            </p>
          </div>

          {/* Project Info */}
          <div className="animate-in slide-in-from-bottom duration-800 delay-400">
            <h4 className="text-lg font-semibold mb-4 transition-colors duration-300 hover:brightness-110" style={{ color: '#C1F21D' }}>Project Focus</h4>
            <ul className="space-y-2 text-white opacity-80">
              <li className="transition-all duration-300 hover:opacity-100 hover:translate-x-2 transform cursor-pointer">• Anonymized Geotracks</li>
              <li className="transition-all duration-300 hover:opacity-100 hover:translate-x-2 transform cursor-pointer">• Privacy Protection</li>
              <li className="transition-all duration-300 hover:opacity-100 hover:translate-x-2 transform cursor-pointer">• Data Analytics</li>
              <li className="transition-all duration-300 hover:opacity-100 hover:translate-x-2 transform cursor-pointer">• Urban Mobility</li>
            </ul>
          </div>

          {/* Event Info */}
          <div className="animate-in slide-in-from-bottom duration-800 delay-600">
            <h4 className="text-lg font-semibold mb-4 transition-colors duration-300 hover:brightness-110" style={{ color: '#C1F21D' }}>Event</h4>
            <div className="text-white opacity-80">
              <p className="mb-2 transition-transform duration-300 hover:scale-105 cursor-pointer">
                <Highlight>Decentrathon 2025</Highlight>
              </p>
              <p className="text-sm transition-opacity duration-300 hover:opacity-100">
                Developing innovative solutions for decentralized mobility and privacy
              </p>
            </div>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t text-center transition-all duration-300 hover:border-opacity-80 animate-in slide-in-from-bottom duration-800 delay-800" style={{ borderColor: '#C1F21D' }}>
          <p className="text-white opacity-60 transition-opacity duration-300 hover:opacity-90">
            🏆 Decentrathon 2025 Winners - Team Richards • Built with privacy in mind
          </p>
        </div>
      </div>
    </footer>
  );
}