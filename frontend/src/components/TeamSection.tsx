export function TeamSection() {
  return (
    <section className="py-20" style={{ backgroundColor: 'rgba(0, 0, 0, 1)' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-6">
          <h2 id="team" className="text-4xl md:text-5xl font-bold mb-6 pt-32 cursor-pointer transition-colors duration-200" 
            style={{ marginTop: '-8rem', color: 'rgba(209, 254, 23, 1)' }}
            onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(180, 220, 20, 1)'}
            onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
          >
            Meet Team Richards
          </h2>
          <p className="text-xl max-w-3xl mx-auto cursor-pointer transition-colors duration-200" 
            style={{ color: 'rgba(247, 247, 248, 1)' }}
            onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
            onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
          >
            Passionate ML engineers and storytellers from Nazarbayev University
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
          {/* Team Members */}
          <div className="text-center p-8 rounded-2xl shadow-sm transition-all duration-200 cursor-pointer group">
            <div className="mb-4 flex justify-center">
              <img 
                src="/dauren.webp" 
                alt="Dauren" 
                className="w-32 h-32 rounded-full object-cover shadow-lg"
              />
            </div>
            <div className="text-2xl font-bold mb-4 transition-colors duration-200" 
              style={{ color: 'rgba(247, 247, 248, 1)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
            >
              Dauren
            </div>
            <a  
              href="https://www.linkedin.com/in/dauren-apas/"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center transition-all duration-200 hover:opacity-80"
            >
              <svg width="32" height="32" viewBox="0 0 382 382" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M347.445,0H34.555C15.471,0,0,15.471,0,34.555v312.889C0,366.529,15.471,382,34.555,382h312.889C366.529,382,382,366.529,382,347.444V34.555C382,15.471,366.529,0,347.445,0z M118.207,329.844c0,5.554-4.502,10.056-10.056,10.056H65.345c-5.554,0-10.056-4.502-10.056-10.056V150.403c0-5.554,4.502-10.056,10.056-10.056h42.806c5.554,0,10.056,4.502,10.056,10.056V329.844z M86.748,123.432c-22.459,0-40.666-18.207-40.666-40.666S64.289,42.1,86.748,42.1s40.666,18.207,40.666,40.666S109.208,123.432,86.748,123.432z M341.91,330.654c0,5.106-4.14,9.246-9.246,9.246H286.73c-5.106,0-9.246-4.14-9.246-9.246v-84.168c0-12.556,3.683-55.021-32.813-55.021c-28.309,0-34.051,29.066-35.204,42.11v97.079c0,5.106-4.139,9.246-9.246,9.246h-44.426c-5.106,0-9.246-4.14-9.246-9.246V149.593c0-5.106,4.14-9.246,9.246-9.246h44.426c5.106,0,9.246,4.14,9.246,9.246v15.655c10.497-15.753,26.097-27.912,59.312-27.912c73.552,0,73.131,68.716,73.131,106.472L341.91,330.654L341.91,330.654z" fill="rgba(209, 254, 23, 1)"/>
              </svg>
            </a>
          </div>
          
          <div className="text-center p-8 rounded-2xl shadow-sm transition-all duration-200 cursor-pointer group animate-in slide-in-from-bottom duration-800 delay-500">
            <div className="mb-4 flex justify-center">
              <img 
                src="/fatikh.webp" 
                alt="Fatikh" 
                className="w-32 h-32 rounded-full object-cover shadow-lg"
              />
            </div>
            <div className="text-2xl font-bold mb-4 transition-colors duration-200" 
              style={{ color: 'rgba(247, 247, 248, 1)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
            >
              Fatikh
            </div>
            <a 
              href="https://www.linkedin.com/in/fatikh-aidyn/?lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BZZZ8Y7quS8C7syeFPRM1xQ%3D%3D"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center transition-all duration-200 hover:opacity-80"
            >
              <svg width="32" height="32" viewBox="0 0 382 382" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M347.445,0H34.555C15.471,0,0,15.471,0,34.555v312.889C0,366.529,15.471,382,34.555,382h312.889C366.529,382,382,366.529,382,347.444V34.555C382,15.471,366.529,0,347.445,0z M118.207,329.844c0,5.554-4.502,10.056-10.056,10.056H65.345c-5.554,0-10.056-4.502-10.056-10.056V150.403c0-5.554,4.502-10.056,10.056-10.056h42.806c5.554,0,10.056,4.502,10.056,10.056V329.844z M86.748,123.432c-22.459,0-40.666-18.207-40.666-40.666S64.289,42.1,86.748,42.1s40.666,18.207,40.666,40.666S109.208,123.432,86.748,123.432z M341.91,330.654c0,5.106-4.14,9.246-9.246,9.246H286.73c-5.106,0-9.246-4.14-9.246-9.246v-84.168c0-12.556,3.683-55.021-32.813-55.021c-28.309,0-34.051,29.066-35.204,42.11v97.079c0,5.106-4.139,9.246-9.246,9.246h-44.426c-5.106,0-9.246-4.14-9.246-9.246V149.593c0-5.106,4.14-9.246,9.246-9.246h44.426c5.106,0,9.246,4.14,9.246,9.246v15.655c10.497-15.753,26.097-27.912,59.312-27.912c73.552,0,73.131,68.716,73.131,106.472L341.91,330.654L341.91,330.654z" fill="rgba(209, 254, 23, 1)"/>
              </svg>
            </a>
          </div>
        </div>

        {/* Call to Action */}
        <div className="text-center animate-in slide-in-from-bottom duration-800 delay-900">
          <div className="p-8 rounded-2xl border-2 transition-all duration-200 hover:border-opacity-80 group">
            <h3 className="text-2xl md:text-3xl font-bold mb-4 cursor-pointer transition-colors duration-200" 
              style={{ color: 'rgba(209, 254, 23, 1)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(180, 220, 20, 1)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
            >
              Interested in our work?
            </h3>
            <p className="text-lg mb-6 cursor-pointer transition-colors duration-200" 
              style={{ color: 'rgba(247, 247, 248, 1)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
            >
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