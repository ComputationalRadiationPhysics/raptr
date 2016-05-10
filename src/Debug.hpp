/** @file Debug.hpp */

#ifndef DEBUG_HPP
#define DEBUG_HPP

// The no debug level
#define RAPTR_DEBUG_DISABLED 0

// The minimal debug level
#define RAPTR_DEBUG_MINIMAL 1

#ifndef RAPTR_DEBUG
  // Set the minimal debug level if no debug level is defined
  #define RAPTR_DEBUG RAPTR_DEBUG_DISABLED
#endif

#endif // DEBUG_HPP
