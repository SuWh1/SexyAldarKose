import { useState } from 'react';

export function HackathonHeader() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const handleNavigation = (section: string) => {
    setIsMobileMenuOpen(false); // Close mobile menu after navigation
    const element = document.getElementById(section);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header className="shadow-sm sticky top-0 z-[1100] transition-all duration-300 relative border-b" style={{ backgroundColor: 'rgba(9, 13, 14, 1)', borderBottomColor: 'rgba(153, 153, 153, 0.2)' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex-shrink-0">
            <div className="flex items-center gap-3 group cursor-pointer">
              <img 
                src="/icon.png" 
                alt="SexyAldarKose" 
                className="h-10 w-10 object-contain"
              />
              <h1 className="text-2xl font-bold transition-colors duration-200 group-hover:text-[rgba(209,254,23,1)]" style={{ color: 'rgba(247, 247, 248, 1)' }}>
                SexyAldarKose
              </h1>
            </div>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex space-x-2 items-center">
            <button 
              onClick={() => handleNavigation('approach')}
              className="px-6 py-3 text-base font-medium transition-colors duration-200 rounded-lg cursor-pointer"
              style={{ color: 'rgba(153, 153, 153, 1)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(153, 153, 153, 1)'}
            >
              Approach
            </button>
            <button 
              onClick={() => handleNavigation('team')}
              className="px-6 py-3 text-base font-medium transition-colors duration-200 rounded-lg cursor-pointer"
              style={{ color: 'rgba(153, 153, 153, 1)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(153, 153, 153, 1)'}
            >
              Team
            </button>
          </nav>

          {/* Desktop CTA */}
          <div className="hidden md:flex">
            <a 
              href="https://higgsfield.ai/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 hover:brightness-90 cursor-pointer" 
              style={{ backgroundColor: 'rgba(209, 254, 23, 1)', color: 'rgba(9, 13, 14, 1)' }}
            >
              Higgsfield AI
            </a>
          </div>

          {/* Mobile hamburger button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="p-2 rounded-md transition-all duration-300"
              style={{ color: 'rgba(153, 153, 153, 1)' }}
              aria-label="Toggle mobile menu"
            >
              <svg
                className={`h-6 w-6 transition-transform duration-300 ${isMobileMenuOpen ? 'rotate-90' : ''}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                {isMobileMenuOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        <div 
          className={`md:hidden absolute top-full left-0 right-0 border-t shadow-lg z-[1200] transition-all duration-300 ease-in-out transform ${
            isMobileMenuOpen 
              ? 'opacity-100 translate-y-0 scale-y-100' 
              : 'opacity-0 -translate-y-2 scale-y-95 pointer-events-none'
          }`}
          style={{ transformOrigin: 'top', backgroundColor: 'rgba(9, 13, 14, 1)', borderColor: 'rgba(153, 153, 153, 0.3)' }}
        >
          <div className="px-2 pt-2 pb-3 space-y-1">
            <button
              onClick={() => handleNavigation('approach')}
              className="block w-full text-left px-3 py-2 text-base font-medium transition-all duration-200 rounded-lg transform hover:translate-x-1"
              style={{ color: 'rgba(153, 153, 153, 1)' }}
            >
              Approach
            </button>
            <button
              onClick={() => handleNavigation('team')}
              className="block w-full text-left px-3 py-2 text-base font-medium transition-all duration-200 rounded-lg transform hover:translate-x-1"
              style={{ color: 'rgba(153, 153, 153, 1)' }}
            >
              Team
            </button>
            
            <a 
              href="https://higgsfield.ai/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="block px-3 py-2 rounded-lg text-base font-medium transition-all duration-200 hover:brightness-95 cursor-pointer transform hover:translate-x-1" 
              style={{ backgroundColor: 'rgba(209, 254, 23, 1)', color: 'rgba(9, 13, 14, 1)' }}
              onClick={() => setIsMobileMenuOpen(false)}
            >
              Higgsfield AI
            </a>
          </div>
        </div>
      </div>
    </header>
  );
}
