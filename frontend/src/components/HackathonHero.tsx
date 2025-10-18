export function HackathonHero() {
  return (
    <section className="py-12 animate-in fade-in duration-1000" style={{ backgroundColor: 'rgba(9, 13, 14, 1)' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          {/* Badge */}
          <div className="mb-8 animate-in slide-in-from-top duration-700 delay-200">
            <a 
              href="https://higgsfield.ai/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-block px-4 py-1.5 rounded-full text-xs sm:text-sm font-semibold transition-all duration-200 hover:brightness-90 cursor-pointer" 
              style={{ backgroundColor: 'rgba(209, 254, 23, 1)', color: 'rgba(9, 13, 14, 1)' }}
            >
              Higgsfield AI Hackathon 2025
            </a>
          </div>

          {/* Main Heading */}
          <h1 className="text-5xl md:text-7xl font-bold mb-8 leading-tight animate-in slide-in-from-bottom duration-800 delay-300" style={{ color: 'rgba(209, 254, 23, 1)' }}>
            <span className="inline-block">
              Aldar Köse
            </span><br />
            <span className="inline-block">
              Storyboard Generator
            </span>
          </h1>

          {/* Subheading */}
          <h2 className="text-2xl md:text-3xl font-semibold mb-12 animate-in slide-in-from-bottom duration-800 delay-500" style={{ color: 'rgba(247, 247, 248, 1)' }}>
            Reimagining Kazakh folklore through AI<br />
            From script to storyboard in seconds
          </h2>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-6 justify-center my-12 animate-in slide-in-from-bottom duration-800 delay-700">
            <a 
              href="#approach"
              className="px-8 py-4 rounded-lg text-xl font-semibold transition-all duration-200 hover:brightness-90 shadow-lg inline-block text-center"
              style={{ backgroundColor: 'rgba(209, 254, 23, 1)', color: 'rgba(9, 13, 14, 1)' }}
            >
              Learn More →
            </a>
            <a 
              href="#team"
              className="px-8 py-4 rounded-lg text-xl font-semibold border-2 transition-all duration-200 hover:brightness-110 inline-block text-center"
              style={{ borderColor: 'rgba(209, 254, 23, 1)', color: 'rgba(209, 254, 23, 1)' }}
            >
              Meet the Team →
            </a>
          </div>

          {/* Description */}
          <p className="text-base md:text-lg max-w-4xl mx-auto leading-relaxed animate-in slide-in-from-bottom duration-800 delay-1000" style={{ color: 'rgba(153, 153, 153, 1)' }}>
            An intelligent system that automatically generates 6-10 frame storyboards from 
            short scripts, bringing Aldar Köse's adventures to life with AI-powered visual storytelling.
          </p>
        </div>
      </div>
    </section>
  );
}