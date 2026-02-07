// ============================================
// Spot the Scam - Interactive Wiki Scripts
// ============================================

document.addEventListener('DOMContentLoaded', () => {
  initializeViewportOffsets();
  initializeScrollProgress();
  initializeScrollAnimations();
  initializeSmoothScroll();
  initializeMetricsCounter();
  initializeMermaid();
  initializeThemeToggle();
  initializeBackToTop();
  initializeCopyButtons();
  initializeImageLightbox();
  initializeActiveNavigation();
  initializeHamburgerMenu();
});

// ============================================
// Viewport and Layout Offsets
// ============================================
function initializeViewportOffsets() {
  const updateNavbarHeight = () => {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;
    document.documentElement.style.setProperty('--navbar-height', `${navbar.offsetHeight}px`);
  };

  updateNavbarHeight();
  window.addEventListener('resize', updateNavbarHeight);
  window.addEventListener('orientationchange', updateNavbarHeight);
}

// ============================================
// Scroll Progress Bar
// ============================================
function initializeScrollProgress() {
  const progressBar = document.getElementById('scroll-progress');
  let ticking = false;
  
  window.addEventListener('scroll', () => {
    if (!ticking) {
      window.requestAnimationFrame(() => {
        const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (window.scrollY / windowHeight) * 100;
        progressBar.style.width = scrolled + '%';
        ticking = false;
      });
      ticking = true;
    }
  });
}

// ============================================
// Scroll Animations
// ============================================
function initializeScrollAnimations() {
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }
    });
  }, observerOptions);

  // Observe all sections and cards
  document.querySelectorAll('.section, .card, .metric-card').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'opacity 0.8s cubic-bezier(0.4, 0, 0.2, 1), transform 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
    observer.observe(el);
  });
}

// ============================================
// Smooth Scroll for Navigation Links
// ============================================
function initializeSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const href = this.getAttribute('href');
      if (href === '#' || !href) return;
      
      e.preventDefault();
      const target = document.querySelector(href);
      if (target) {
        const offsetTop = target.offsetTop - 80; // Account for fixed navbar
        window.scrollTo({
          top: offsetTop,
          behavior: 'smooth'
        });
        
        // Close mobile menu after clicking
        const navLinks = document.querySelector('.navbar-links');
        const navToggle = document.querySelector('.navbar-toggle');
        const overlay = document.querySelector('.navbar-overlay');
        if (navLinks && navLinks.classList.contains('active')) {
          navLinks.classList.remove('active');
          navToggle.classList.remove('active');
          if (overlay) overlay.classList.remove('active');
          document.body.style.overflow = '';
        }
      }
    });
  });
}

// ============================================
// Animated Counter for Metrics
// ============================================
function initializeMetricsCounter() {
  const counters = document.querySelectorAll('.metric-value');
  const speed = 200; // Animation speed

  const animateCounter = (counter) => {
    const target = parseFloat(counter.getAttribute('data-target'));
    const increment = target / speed;
    let current = 0;

    const updateCounter = () => {
      current += increment;
      if (current < target) {
        // Format based on whether it's a percentage or decimal
        if (target >= 1) {
          counter.textContent = Math.ceil(current) + '%';
        } else {
          counter.textContent = current.toFixed(3);
        }
        requestAnimationFrame(updateCounter);
      } else {
        if (target >= 1) {
          counter.textContent = target + '%';
        } else {
          counter.textContent = target.toFixed(3);
        }
      }
    };

    updateCounter();
  };

  const observerOptions = {
    threshold: 0.5
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
        entry.target.classList.add('counted');
        animateCounter(entry.target);
      }
    });
  }, observerOptions);

  counters.forEach(counter => observer.observe(counter));
}

// ============================================
// Initialize Mermaid Diagrams
// ============================================
function initializeMermaid() {
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'dark',
      themeVariables: {
        primaryColor: '#2563eb',
        primaryTextColor: '#f8fafc',
        primaryBorderColor: '#334155',
        lineColor: '#64748b',
        secondaryColor: '#10b981',
        tertiaryColor: '#f59e0b',
        background: '#1e293b',
        mainBkg: '#1e293b',
        secondBkg: '#0f172a',
        border1: '#334155',
        border2: '#475569',
        note: '#1e293b',
        noteBkg: '#334155',
        noteText: '#f8fafc',
        text: '#f8fafc',
        critical: '#ef4444',
        done: '#10b981',
        active: '#2563eb',
        grid: '#334155',
        nodeBorder: '#475569',
        clusterBkg: '#1e293b',
        clusterBorder: '#475569',
        titleColor: '#f8fafc',
        edgeLabelBackground: '#0f172a',
        actorBorder: '#475569',
        actorBkg: '#1e293b',
        actorTextColor: '#f8fafc',
        actorLineColor: '#64748b',
        signalColor: '#f8fafc',
        signalTextColor: '#f8fafc',
        labelBoxBkgColor: '#1e293b',
        labelBoxBorderColor: '#475569',
        labelTextColor: '#f8fafc',
        loopTextColor: '#f8fafc',
        noteBorderColor: '#475569',
        activationBorderColor: '#475569',
        activationBkgColor: '#334155',
        sequenceNumberColor: '#f8fafc'
      },
      flowchart: {
        curve: 'basis',
        padding: 20
      },
      sequence: {
        diagramMarginX: 50,
        diagramMarginY: 30,
        actorMargin: 50,
        width: 150,
        height: 65,
        boxMargin: 10,
        boxTextMargin: 5,
        noteMargin: 10,
        messageMargin: 35
      }
    });
  }
}

// ============================================
// Theme Toggle (Optional Enhancement)
// ============================================
function initializeThemeToggle() {
  const themeToggle = document.getElementById('theme-toggle');
  if (!themeToggle) return;

  themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('light-mode');
    const icon = themeToggle.querySelector('.badge-icon');
    icon.textContent = document.body.classList.contains('light-mode') ? '🌙' : '☀️';
  });
}

// ============================================
// Back to Top Button
// ============================================
function initializeBackToTop() {
  const backToTop = document.createElement('button');
  backToTop.id = 'back-to-top';
  backToTop.innerHTML = '↑';
  backToTop.style.cssText = `
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: var(--primary-color);
    color: white;
    border: none;
    cursor: pointer;
    opacity: 0;
    transition: all 0.3s ease;
    z-index: 999;
    font-size: 1.5rem;
    box-shadow: var(--shadow-lg);
  `;
  document.body.appendChild(backToTop);

  window.addEventListener('scroll', () => {
    if (window.scrollY > 500) {
      backToTop.style.opacity = '1';
    } else {
      backToTop.style.opacity = '0';
    }
  });

  backToTop.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  backToTop.addEventListener('mouseenter', () => {
    backToTop.style.transform = 'translateY(-5px)';
    backToTop.style.boxShadow = 'var(--shadow-xl)';
  });

  backToTop.addEventListener('mouseleave', () => {
    backToTop.style.transform = 'translateY(0)';
    backToTop.style.boxShadow = 'var(--shadow-lg)';
  });
}

// ============================================
// Copy Code Buttons
// ============================================
function initializeCopyButtons() {
  document.querySelectorAll('pre').forEach((block) => {
    const button = document.createElement('button');
    button.className = 'copy-btn';
    button.textContent = 'Copy';
    button.style.cssText = `
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      padding: 0.5rem 1rem;
      background: var(--primary-color);
      color: white;
      border: none;
      border-radius: 0.5rem;
      cursor: pointer;
      font-size: 0.8rem;
      font-weight: 600;
      opacity: 0;
      transition: all 0.3s ease;
    `;

    block.style.position = 'relative';
    block.appendChild(button);

    block.addEventListener('mouseenter', () => {
      button.style.opacity = '1';
    });

    block.addEventListener('mouseleave', () => {
      button.style.opacity = '0';
    });

    button.addEventListener('click', () => {
      const code = block.querySelector('code')?.textContent || block.textContent;
      navigator.clipboard.writeText(code).then(() => {
        button.textContent = 'Copied!';
        button.style.background = 'var(--secondary-color)';
        setTimeout(() => {
          button.textContent = 'Copy';
          button.style.background = 'var(--primary-color)';
        }, 2000);
      });
    });
  });
}

// ============================================
// Image Lightbox
// ============================================
function initializeImageLightbox() {
  const lightbox = document.querySelector('.lightbox');
  if (!lightbox) return;

  const lightboxImage = lightbox.querySelector('img');
  const closeButton = lightbox.querySelector('.lightbox-close');
  let lastFocusedElement = null;

  const openLightbox = (img) => {
    lastFocusedElement = document.activeElement;
    const source = img.currentSrc || img.src;
    lightboxImage.src = source;
    lightboxImage.alt = img.alt || 'Expanded image preview';
    lightbox.classList.add('active');
    lightbox.setAttribute('aria-hidden', 'false');
    document.body.classList.add('lightbox-open');
    closeButton.focus();
  };

  const closeLightbox = () => {
    lightbox.classList.remove('active');
    lightbox.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('lightbox-open');
    lightboxImage.src = '';
    lightboxImage.alt = '';
    if (lastFocusedElement) {
      lastFocusedElement.focus();
    }
  };

  document.querySelectorAll('.screenshot img').forEach((img) => {
    img.setAttribute('role', 'button');
    img.setAttribute('tabindex', '0');
    img.addEventListener('click', () => openLightbox(img));
    img.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        openLightbox(img);
      }
    });
  });

  closeButton.addEventListener('click', closeLightbox);

  lightbox.addEventListener('click', (event) => {
    if (event.target === lightbox) {
      closeLightbox();
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && lightbox.classList.contains('active')) {
      closeLightbox();
    }
  });
}

// ============================================
// Card Hover Effects (Disabled)
// ============================================

// ============================================
// Active Navigation Link Highlighting
// ============================================
function initializeActiveNavigation() {
  const sections = document.querySelectorAll('.section, .hero');
  const navLinks = document.querySelectorAll('.navbar-links a[href^="#"]');
  
  // Create a map of section IDs to nav links
  const sectionMap = new Map();
  navLinks.forEach(link => {
    const href = link.getAttribute('href');
    if (href && href !== '#') {
      sectionMap.set(href.substring(1), link);
    }
  });

  const observerOptions = {
    rootMargin: '-100px 0px -66%',
    threshold: 0
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      const id = entry.target.id;
      const link = sectionMap.get(id);
      
      if (entry.isIntersecting && link) {
        // Remove active class from all links
        navLinks.forEach(l => l.classList.remove('active'));
        // Add active class to current link
        link.classList.add('active');
      }
    });
  }, observerOptions);

  sections.forEach(section => {
    if (section.id) {
      observer.observe(section);
    }
  });
}

// ============================================
// Hamburger Menu Toggle
// ============================================
function initializeHamburgerMenu() {
  const navToggle = document.querySelector('.navbar-toggle');
  const navLinks = document.querySelector('.navbar-links');
  
  if (!navToggle || !navLinks) return;

  // Create overlay element
  const overlay = document.createElement('div');
  overlay.className = 'navbar-overlay';
  document.body.appendChild(overlay);

  const toggleMenu = (forceClose = false) => {
    if (forceClose) {
      navToggle.classList.remove('active');
      navLinks.classList.remove('active');
      overlay.classList.remove('active');
      document.body.style.overflow = '';
    } else {
      navToggle.classList.toggle('active');
      navLinks.classList.toggle('active');
      overlay.classList.toggle('active');
      
      const mediaQuery = window.matchMedia('(max-width: 1200px)');
      if (mediaQuery.matches && navLinks.classList.contains('active')) {
        document.body.style.overflow = 'hidden';
      } else {
        document.body.style.overflow = '';
      }
    }
  };

  // Toggle menu on button click
  navToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleMenu();
  });

  // Close menu when clicking overlay
  overlay.addEventListener('click', () => {
    toggleMenu(true);
  });

  // Close menu when clicking outside
  document.addEventListener('click', (e) => {
    if (!navToggle.contains(e.target) && 
        !navLinks.contains(e.target) && 
        navLinks.classList.contains('active')) {
      toggleMenu(true);
    }
  });

  // Close menu on escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && navLinks.classList.contains('active')) {
      toggleMenu(true);
    }
  });

  // Close menu on window resize if screen gets larger
  window.addEventListener('resize', () => {
    const mediaQuery = window.matchMedia('(max-width: 1200px)');
    if (!mediaQuery.matches) {
      toggleMenu(true);
    }
  });

  // Close menu when navigation link is clicked (handled in smooth scroll)
}

// ============================================
// Navbar Scroll Effect
// ============================================
window.addEventListener('scroll', () => {
  const navbar = document.querySelector('.navbar');
  if (window.scrollY > 50) {
    navbar.style.background = 'rgba(15, 23, 42, 0.95)';
    navbar.style.boxShadow = 'var(--shadow-lg)';
  } else {
    navbar.style.background = 'rgba(15, 23, 42, 0.8)';
    navbar.style.boxShadow = 'var(--shadow-md)';
  }
});

// ============================================
// Typing Animation for Hero Subtitle
// ============================================
function typeWriter(element, text, speed = 50) {
  let i = 0;
  element.textContent = '';
  
  function type() {
    if (i < text.length) {
      element.textContent += text.charAt(i);
      i++;
      setTimeout(type, speed);
    }
  }
  
  type();
}

// ============================================
// Performance Stats Live Update (Demo)
// ============================================
function initializeLiveStats() {
  const stats = document.querySelectorAll('.metric-value');
  
  // Simulate real-time updates
  setInterval(() => {
    stats.forEach(stat => {
      const current = parseFloat(stat.textContent);
      const variance = (Math.random() - 0.5) * 0.01; // Small random variance
      const newValue = current + variance;
      
      if (current >= 1) {
        stat.textContent = newValue.toFixed(1) + '%';
      } else {
        stat.textContent = newValue.toFixed(3);
      }
    });
  }, 5000);
}

// ============================================
// Easter Egg: Konami Code
// ============================================
let konamiCode = [];
const konamiSequence = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];

document.addEventListener('keydown', (e) => {
  konamiCode.push(e.key);
  konamiCode = konamiCode.slice(-10);
  
  if (konamiCode.join(',') === konamiSequence.join(',')) {
    activateEasterEgg();
  }
});

function activateEasterEgg() {
  document.body.style.animation = 'rainbow 2s linear infinite';
  
  const style = document.createElement('style');
  style.textContent = `
    @keyframes rainbow {
      0% { filter: hue-rotate(0deg); }
      100% { filter: hue-rotate(360deg); }
    }
  `;
  document.head.appendChild(style);
  
  setTimeout(() => {
    document.body.style.animation = '';
    style.remove();
  }, 10000);
  
  alert('🎉 You found the easter egg! Enjoy the rainbow! 🌈');
}

// ============================================
// Analytics (Optional - Add your tracking)
// ============================================
function trackEvent(category, action, label) {
  // Placeholder for analytics tracking
  console.log('Event tracked:', { category, action, label });
  
  // Example: Google Analytics
  // if (typeof gtag !== 'undefined') {
  //   gtag('event', action, {
  //     'event_category': category,
  //     'event_label': label
  //   });
  // }
}

// Track CTA clicks
document.querySelectorAll('.btn-primary, .btn-secondary').forEach(btn => {
  btn.addEventListener('click', (e) => {
    const label = e.target.textContent.trim();
    trackEvent('CTA', 'click', label);
  });
});

// ============================================
// Export functions for external use
// ============================================
window.spotTheScam = {
  trackEvent,
  typeWriter,
  initializeLiveStats
};

console.log('%c🛡️ Spot the Scam Wiki Loaded Successfully!', 'color: #2563eb; font-size: 16px; font-weight: bold;');
console.log('%cDetecting fraud with AI precision 🤖', 'color: #10b981; font-size: 14px;');
