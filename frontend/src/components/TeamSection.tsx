import { Highlight } from './Highlight';

export function TeamSection() {
  return (
    <section className="py-20" style={{ backgroundColor: '#FFFEEB' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-6">
          <h2 id="team" className="text-4xl md:text-5xl font-bold mb-6 pt-32 animate-in slide-in-from-bottom duration-800 transition-transform hover:scale-105" style={{ color: '#141414', marginTop: '-8rem' }}>
            Meet Team <Highlight>Richards</Highlight>
          </h2>
          <p className="text-xl max-w-3xl mx-auto animate-in slide-in-from-bottom duration-800 delay-200 hover:opacity-100 transition-opacity" style={{ color: '#141414', opacity: 0.8 }}>
            A dedicated team of developers and privacy experts working on 
            <Highlight>innovative geotrack solutions</Highlight> for Decentrathon 2025
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
          {/* Team Stats */}
          <div className="text-center p-8 bg-white rounded-2xl shadow-sm transition-all duration-300 hover:scale-105 hover:shadow-xl transform cursor-pointer group animate-in slide-in-from-bottom duration-800 delay-300">
            <div className="text-4xl font-bold mb-2 transition-all duration-500 group-hover:scale-110 group-hover:text-5xl" style={{ color: '#C1F21D' }}>
              100%
            </div>
            <p className="transition-colors duration-300 group-hover:opacity-100" style={{ color: '#141414', opacity: 0.8 }}>
              Privacy Focused
            </p>
          </div>
          
          <div className="text-center p-8 bg-white rounded-2xl shadow-sm transition-all duration-300 hover:scale-105 hover:shadow-xl transform cursor-pointer group animate-in slide-in-from-bottom duration-800 delay-500">
            <div className="text-4xl font-bold mb-2 transition-all duration-500 group-hover:scale-110 group-hover:text-5xl" style={{ color: '#C1F21D' }}>
              0
            </div>
            <p className="transition-colors duration-300 group-hover:opacity-100" style={{ color: '#141414', opacity: 0.8 }}>
              Personal Data Exposed
            </p>
          </div>
          
          <div className="text-center p-8 bg-white rounded-2xl shadow-sm transition-all duration-300 hover:scale-105 hover:shadow-xl transform cursor-pointer group animate-in slide-in-from-bottom duration-800 delay-700">
            <div className="text-4xl font-bold mb-2 transition-all duration-500 group-hover:scale-110 group-hover:text-5xl group-hover:rotate-12" style={{ color: '#C1F21D' }}>
              ∞
            </div>
            <p className="transition-colors duration-300 group-hover:opacity-100" style={{ color: '#141414', opacity: 0.8 }}>
              Possibilities
            </p>
          </div>
        </div>

        {/* Call to Action */}
        <div className="text-center animate-in slide-in-from-bottom duration-800 delay-900">
          <div className="p-8 rounded-2xl transition-all duration-300 hover:shadow-xl transform hover:scale-[1.02] group" style={{ backgroundColor: '#C1F21D' }}>
            <h3 className="text-2xl md:text-3xl font-bold mb-4 transition-transform duration-300 group-hover:scale-105" style={{ color: '#141414' }}>
              Interested in our <Highlight>work</Highlight>?
            </h3>
            <p className="text-lg mb-6 transition-opacity duration-300 group-hover:opacity-100" style={{ color: '#141414', opacity: 0.8 }}>
              Discover how we're revolutionizing geotrack analytics while protecting privacy
            </p>
            <a 
              href="https://decentrathon.ai/"
              target="_blank"
              rel="noopener noreferrer"
              className="px-8 py-4 rounded-lg text-lg font-semibold text-white transition-all duration-300 hover:scale-105 hover:shadow-lg transform group-button inline-block"
              style={{ backgroundColor: '#141414' }}
            >
              <span className="transition-transform duration-300 group-button-hover:translate-x-1 inline-block">
                Learn More →
              </span>
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}