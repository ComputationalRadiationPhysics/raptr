/**
 * Copyright 2016 Malte Zacharias
 *
 * This file is part of raptr.
 *
 * raptr is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * raptr is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with raptr.
 * If not, see <http://www.gnu.org/licenses/>.
 */

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

