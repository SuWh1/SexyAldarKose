export function AldarKoseComics() {
  // Get all images from the public/aldarKose folder
  const images = [
    '/aldarKose/{0A425703-CCDC-49D2-B405-9AF92E497D9F}.png',
    '/aldarKose/{0C7BDE00-EA22-4A68-A754-EC07F987E5F9}.png',
    '/aldarKose/{28C35647-E798-4E52-9DD4-4B9110704B37}.png',
    '/aldarKose/{317A7AC9-4D29-44F1-B35A-73584F30D2F4}.png',
    '/aldarKose/{3466E0CC-A50F-4994-9185-B8C6D54B0D0D}.png',
    '/aldarKose/{3DC4CAC8-98DB-44FA-AB4E-04541E831127}.png',
    '/aldarKose/{47D38B95-90CD-4951-B505-44CE3EC4167C}.png',
    '/aldarKose/{53C28B83-4BA6-422A-A181-BF2C6446039E}.png',
    '/aldarKose/{692ED769-C77B-4F2B-BFD8-088E0C614908}.png',
    '/aldarKose/{92260B0B-84AC-46D9-9CFE-C516CDB22503}.png',
  ];

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'rgba(0, 0, 0, 1)' }}>
      <section className="py-12 duration-1000">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Title */}
          <div className="text-center mb-12">
            <h1 
              className="text-5xl md:text-7xl font-bold mb-6 leading-tight transition-colors duration-200" 
              style={{ color: 'rgba(209, 254, 23, 1)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(180, 220, 20, 1)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
            >
              Aldar Köse Comics
            </h1>
          </div>

          {/* Comics Panel - One image per line, center aligned */}
          <div className="flex flex-col items-center gap-8 max-w-2xl mx-auto">
            {images.map((image, index) => (
              <div 
                key={index}
                className="w-full"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <img 
                  src={image} 
                  alt={`Aldar Köse panel ${index + 1}`}
                  className="w-full h-auto rounded-lg shadow-2xl"
                  style={{ 
                    border: '2px solid rgba(27, 29, 17, 1)',
                  }}
                  loading={index > 2 ? "lazy" : "eager"}
                />
              </div>
            ))}
          </div>

          {/* Back Button */}
          <div className="flex justify-center mt-12">
            <a 
              href="/"
              className="px-8 py-4 rounded-2xl text-lg font-semibold transition-all duration-200 hover:brightness-90 shadow-lg whitespace-nowrap flex items-center gap-2"
              style={{ backgroundColor: 'rgba(209, 254, 23, 1)', color: 'rgba(9, 13, 14, 1)' }}
            >
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 5L7 10L12 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Back to Home
            </a>
          </div>
        </div>
      </section>
    </div>
  );
}
