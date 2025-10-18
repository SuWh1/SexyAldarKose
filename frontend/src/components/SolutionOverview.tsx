import { Highlight } from './Highlight';

export function SolutionOverview() {
  return (
    <section id="solution" className="py-20" style={{ backgroundColor: '#141414' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          {/* Left Content */}
          <div className="animate-in slide-in-from-left duration-800">
            <h2 id="approach" className="text-4xl md:text-5xl font-bold mb-8 leading-tight text-white pt-40 transition-transform duration-500 hover:scale-105" style={{ marginTop: '-13rem' }}>
              Our <Highlight>Approach</Highlight> to<br />
              Privacy Protection
            </h2>
            
            <div className="space-y-6 text-lg text-white opacity-90">
              <p className="animate-in slide-in-from-left duration-800 delay-200 hover:opacity-100 transition-opacity duration-300">
                We develop cutting-edge solutions that analyze <Highlight>anonymized geotracks</Highlight> 
                to provide valuable insights while maintaining the highest standards of privacy protection.
              </p>
              
              <p className="animate-in slide-in-from-left duration-800 delay-400 hover:opacity-100 transition-opacity duration-300">
                Our approach ensures that <Highlight>no personal information</Highlight> is exposed, 
                using advanced anonymization techniques that comply with global privacy standards.
              </p>
              
              <p className="animate-in slide-in-from-left duration-800 delay-600 hover:opacity-100 transition-opacity duration-300">
                The solution focuses on extracting meaningful patterns from mobility data 
                to improve urban planning and transportation efficiency.
              </p>
            </div>
          </div>

          {/* Right Content - Key Features */}
          <div className="space-y-6">
            <div className="p-6 rounded-xl transition-all duration-300 hover:scale-105 hover:shadow-xl transform cursor-pointer group animate-in slide-in-from-right duration-800 delay-200" style={{ backgroundColor: '#FFFEEB' }}>
              <div className="text-3xl mb-4 transition-transform duration-300 group-hover:scale-110 group-hover:rotate-12 transform">🔒</div>
              <h3 className="text-xl font-bold mb-3 transition-colors duration-300 group-hover:opacity-80" style={{ color: '#141414' }}>
                <Highlight>Privacy First</Highlight>
              </h3>
              <p className="transition-colors duration-300 group-hover:opacity-90" style={{ color: '#141414', opacity: 0.8 }}>
                Advanced anonymization methods ensure complete user privacy protection
              </p>
            </div>

            <div className="p-6 rounded-xl transition-all duration-300 hover:scale-105 hover:shadow-xl transform cursor-pointer group animate-in slide-in-from-right duration-800 delay-400" style={{ backgroundColor: '#C1F21D' }}>
              <div className="text-3xl mb-4 transition-transform duration-300 group-hover:scale-110 group-hover:rotate-12 transform">📊</div>
              <h3 className="text-xl font-bold mb-3 transition-colors duration-300 group-hover:opacity-80" style={{ color: '#141414' }}>
                Data Analytics
              </h3>
              <p className="transition-colors duration-300 group-hover:opacity-90" style={{ color: '#141414', opacity: 0.8 }}>
                Powerful analytics on anonymized geotracks for actionable insights
              </p>
            </div>

            <div className="p-6 rounded-xl transition-all duration-300 hover:scale-105 hover:shadow-xl transform cursor-pointer group animate-in slide-in-from-right duration-800 delay-600" style={{ backgroundColor: '#FFFEEB' }}>
              <div className="text-3xl mb-4 transition-transform duration-300 group-hover:scale-110 group-hover:rotate-12 transform">🌍</div>
              <h3 className="text-xl font-bold mb-3 transition-colors duration-300 group-hover:opacity-80" style={{ color: '#141414' }}>
                <Highlight>Real Impact</Highlight>
              </h3>
              <p className="transition-colors duration-300 group-hover:opacity-90" style={{ color: '#141414', opacity: 0.8 }}>
                Solutions that improve mobility and urban planning decisions
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}