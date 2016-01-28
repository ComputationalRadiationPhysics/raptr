/** @file CUSPARSE_HandleError.hpp
 * 
 *  @brief Auxiliary macro for printing CUSPARSE error messages.
 */
#ifndef CUSPARSE_HANDLEERROR_HPP
#define	CUSPARSE_HANDLEERROR_HPP

#include <iostream>
#include <cusparse.h>

void getCusparseErrorOutput( cusparseStatus_t err, char const * file,
                                int line ) {
    std::cerr << file << "(" << line << "): cusparse error"
              << std::endl;
}

#define HANDLE_CUSPARSE_ERROR( err ) \
if(err != CUSPARSE_STATUS_SUCCESS) { \
    getCusparseErrorOutput(err, __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
}

#endif	/* CUSPARSE_HANDLEERROR_HPP */

