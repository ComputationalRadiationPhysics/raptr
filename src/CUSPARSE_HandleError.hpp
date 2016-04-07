/** @file CUSPARSE_HandleError.hpp
 * 
 *  @brief Header file that defines an auxiliary macro for printing cuSPARSE
 *  error messages.
 */
#ifndef CUSPARSE_HANDLEERROR_HPP
#define	CUSPARSE_HANDLEERROR_HPP

#include <iostream>
#include <cusparse.h>

/**
 * @brief Print cuSPARSE error line to stderr.
 */
void getCusparseErrorOutput( cusparseStatus_t err, char const * file,
                                int line ) {
    std::cerr << file << "(" << line << "): cusparse error"
              << std::endl;
}

/**
 * @brief Macro to wrap cuSPARSE calls with that invokes cuSPARSE error
 * printing function if necessary.
 */
#define HANDLE_CUSPARSE_ERROR( err ) \
if(err != CUSPARSE_STATUS_SUCCESS) { \
    getCusparseErrorOutput(err, __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
}

#endif	/* CUSPARSE_HANDLEERROR_HPP */

