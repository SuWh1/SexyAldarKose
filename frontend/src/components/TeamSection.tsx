export function TeamSection() {
  return (
    <section className="py-20" style={{ backgroundColor: 'rgba(9, 13, 14, 1)' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-6">
          <h2 id="team" className="text-4xl md:text-5xl font-bold mb-6 pt-32 animate-in slide-in-from-bottom duration-800" style={{ marginTop: '-8rem', color: 'rgba(209, 254, 23, 1)' }}>
            Meet Team Richards
          </h2>
          <p className="text-xl max-w-3xl mx-auto animate-in slide-in-from-bottom duration-800 delay-200" style={{ color: 'rgba(247, 247, 248, 1)' }}>
            Passionate ML engineers and storytellers from Nazarbayev University
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
          {/* Team Members */}
          <div className="text-center p-8 rounded-2xl shadow-sm transition-all duration-200 cursor-pointer group animate-in slide-in-from-bottom duration-800 delay-300" style={{ backgroundColor: 'rgba(9, 13, 14, 1)' }}>
            <div className="mb-4 flex justify-center">
              <img 
                src="/dauren.webp" 
                alt="Dauren" 
                className="w-32 h-32 rounded-full object-cover shadow-lg"
              />
            </div>
            <div className="text-2xl font-bold mb-2" style={{ color: 'rgba(209, 254, 23, 1)' }}>
              Dauren
            </div>
            <p style={{ color: 'rgba(247, 247, 248, 1)' }}>
              Genius
            </p>
          </div>
          
          <div className="text-center p-8 rounded-2xl shadow-sm transition-all duration-200 cursor-pointer group animate-in slide-in-from-bottom duration-800 delay-500" style={{ backgroundColor: 'rgba(9, 13, 14, 1)' }}>
            <div className="mb-4 flex justify-center">
              <img 
                src="/fatikh.webp" 
                alt="Fatikh" 
                className="w-32 h-32 rounded-full object-cover shadow-lg"
              />
            </div>
            <div className="text-2xl font-bold mb-2" style={{ color: 'rgba(209, 254, 23, 1)' }}>
              Fatikh
            </div>
            <p style={{ color: 'rgba(247, 247, 248, 1)' }}>
              Millionaire
            </p>
          </div>
        </div>

        {/* Call to Action */}
        <div className="text-center animate-in slide-in-from-bottom duration-800 delay-900">
          <div className="p-8 rounded-2xl border-2 transition-all duration-200 hover:border-opacity-80 group" style={{ backgroundColor: 'rgba(9, 13, 14, 1)', borderColor: 'rgba(209, 254, 23, 1)' }}>
            <h3 className="text-2xl md:text-3xl font-bold mb-4" style={{ color: 'rgba(209, 254, 23, 1)' }}>
              Interested in our work?
            </h3>
            <p className="text-lg mb-6" style={{ color: 'rgba(247, 247, 248, 1)' }}>
              Discover how we're reviving cultural storytelling with AI-powered creativity
            </p>
            <a 
              href="https://github.com/SuWh1/SexyAldarKose"
              target="_blank"
              rel="noopener noreferrer"
              className="px-8 py-4 rounded-lg text-lg font-semibold transition-all duration-200 hover:brightness-90 inline-block"
              style={{ backgroundColor: 'rgba(209, 254, 23, 1)', color: 'rgba(9, 13, 14, 1)' }}
            >
              View on GitHub →
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}