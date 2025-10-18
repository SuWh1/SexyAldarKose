import { Highlight } from './Highlight';

export function HackathonHero() {
  return (
    <section className="py-12 animate-in fade-in duration-1000" style={{ backgroundColor: '#FFFEEB' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          {/* Badge */}
          <div className="mb-8 animate-in slide-in-from-top duration-700 delay-200">
            <a 
              href="https://decentrathon.ai/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-block px-3 py-2 sm:px-4 sm:py-2 md:px-6 md:py-3 rounded-full text-sm sm:text-base md:text-lg font-semibold transition-all duration-300 hover:scale-105 hover:shadow-lg transform cursor-pointer group" 
              style={{ backgroundColor: '#C1F21D', color: '#141414' }}
            >
              <span className="transition-transform duration-300 group-hover:scale-110 inline-block">🏆</span> 
              <span className="hidden sm:inline">Decentrathon 2025 - Team Richards</span>
              <span className="sm:hidden">Decentrathon 2025</span>
            </a>
          </div>

          {/* Main Heading */}
          <h1 className="text-5xl md:text-7xl font-bold mb-8 leading-tight animate-in slide-in-from-bottom duration-800 delay-300" style={{ color: '#141414' }}>
            <span className="inline-block transition-transform duration-500 hover:scale-105">
              <Highlight>Privacy-First</Highlight>
            </span><br />
            <span className="inline-block transition-transform duration-500 hover:scale-105 delay-100">
              Geotrack Analytics
            </span>
          </h1>

          {/* Subheading */}
          <h2 className="text-2xl md:text-3xl font-semibold mb-12 animate-in slide-in-from-bottom duration-800 delay-500" style={{ color: '#141414', opacity: 0.8 }}>
            Developing solutions based on <Highlight>anonymized geotracks</Highlight><br />
            while protecting user privacy
          </h2>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-6 justify-center my-12 animate-in slide-in-from-bottom duration-800 delay-700">
            <a 
              href="#approach"
              className="px-8 py-4 rounded-lg text-xl font-semibold transition-all duration-300 hover:scale-105 hover:shadow-xl transform hover:brightness-95 shadow-lg group inline-block text-center"
              style={{ backgroundColor: '#C1F21D', color: '#141414' }}
            >
              <span className="transition-transform duration-300 group-hover:translate-x-1 inline-block">
                Learn More →
              </span>
            </a>
            <a 
              href="#team"
              className="px-8 py-4 rounded-lg text-xl font-semibold border-2 transition-all duration-300 hover:scale-105 hover:shadow-lg hover:bg-gray-50 transform group inline-block text-center"
              style={{ borderColor: '#141414', color: '#141414' }}
            >
              <span className="transition-transform duration-300 group-hover:translate-x-1 inline-block">
                Meet the Team →
              </span>
            </a>
          </div>

          {/* Description */}
          <p className="text-xl md:text-2xl max-w-4xl mx-auto leading-relaxed animate-in slide-in-from-bottom duration-800 delay-1000" style={{ color: '#141414', opacity: 0.7 }}>
            Our hackathon solution leverages <Highlight>advanced anonymization methods</Highlight> to analyze 
            mobility patterns without compromising personal information or user privacy.
          </p>
        </div>
      </div>
    </section>
  );
}