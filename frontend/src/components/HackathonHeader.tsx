import { Highlight } from './Highlight';
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
    <header className="bg-white shadow-sm sticky top-0 z-[1100] transition-all duration-300 hover:shadow-md relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex-shrink-0">
            <div className="block group">
              <h1 className="text-2xl font-bold transition-all duration-300 group-hover:scale-105 transform cursor-pointer" style={{ color: '#111111' }}>
                Team <span style={{ color: '#C1F21D' }} className="transition-all duration-300 group-hover:brightness-110">Richards</span>
              </h1>
            </div>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex space-x-2 items-center">
            <button 
              onClick={() => handleNavigation('approach')}
              className="px-4 py-2 text-sm font-medium transition-all duration-300 hover:scale-105 hover:bg-gray-50 rounded-lg transform hover:shadow-sm cursor-pointer" 
              style={{ color: '#111111' }}
            >
              Approach
            </button>
            <button 
              onClick={() => handleNavigation('team')}
              className="px-4 py-2 text-sm font-medium transition-all duration-300 hover:scale-105 hover:bg-gray-50 rounded-lg transform hover:shadow-sm cursor-pointer" 
              style={{ color: '#111111' }}
            >
              Team
            </button>
          </nav>

          {/* Desktop CTA */}
          <div className="hidden md:flex">
            <a 
              href="https://decentrathon.ai/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 hover:scale-105 hover:shadow-lg transform hover:brightness-95 cursor-pointer group" 
              style={{ backgroundColor: '#C1F21D', color: '#111111' }}
            >
              <Highlight>Decentrathon 2025</Highlight>
            </a>
          </div>

          {/* Mobile hamburger button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-all duration-300"
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
          className={`md:hidden absolute top-full left-0 right-0 bg-white border-t border-gray-200 shadow-lg z-[1200] transition-all duration-300 ease-in-out transform ${
            isMobileMenuOpen 
              ? 'opacity-100 translate-y-0 scale-y-100' 
              : 'opacity-0 -translate-y-2 scale-y-95 pointer-events-none'
          }`}
          style={{ transformOrigin: 'top' }}
        >
          <div className="px-2 pt-2 pb-3 space-y-1">
            <button
              onClick={() => handleNavigation('approach')}
              className="block w-full text-left px-3 py-2 text-base font-medium transition-all duration-200 hover:bg-gray-50 rounded-lg transform hover:translate-x-1"
              style={{ color: '#111111' }}
            >
              Approach
            </button>
            <button
              onClick={() => handleNavigation('team')}
              className="block w-full text-left px-3 py-2 text-base font-medium transition-all duration-200 hover:bg-gray-50 rounded-lg transform hover:translate-x-1"
              style={{ color: '#111111' }}
            >
              Team
            </button>
            
            <a 
              href="https://decentrathon.ai/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="block px-3 py-2 rounded-lg text-base font-medium transition-all duration-200 hover:brightness-95 cursor-pointer transform hover:translate-x-1" 
              style={{ backgroundColor: '#C1F21D', color: '#111111' }}
              onClick={() => setIsMobileMenuOpen(false)}
            >
              <Highlight>Decentrathon 2025</Highlight>
            </a>
          </div>
        </div>
      </div>
    </header>
  );
}
