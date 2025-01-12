// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_LAPACK_API_COMMON_HH
#define SLATE_LAPACK_API_COMMON_HH

// get BLAS_FORTRAN_NAME and blas_int
#include "blas/fortran.h"

#include "slate/slate.hh"

//==============================================================================
namespace slate {
namespace lapack_api {

#define logprintf(fmt, ...) \
    do { \
        fprintf( stdout, "%s:%d %s(): " fmt, \
                 __FILE__, __LINE__, __func__, __VA_ARGS__ ); \
        fflush(0); \
    } while (0)

inline char to_char( float*  a ) { return 's'; }
inline char to_char( double* a ) { return 'd'; }
inline char to_char( std::complex<float>*  a ) { return 'c'; }
inline char to_char( std::complex<double>* a ) { return 'z'; }

//==============================================================================
/// Initialize target setting from environment variable.
/// Uses thread-safe Scott Meyers Singleton.
class TargetConfig
{
public:
    /// @return target (HostTask, Devices, etc.) to use.
    static slate::Target value()
    {
        return instance().target_;
    }

    /// Set target to use.
    static void value( slate::Target target )
    {
        instance().target_ = target;
    }

private:
    /// On first call, creates the singleton instance, which queries the
    /// environment variable.
    /// @return singleton instance.
    static TargetConfig& instance()
    {
        static TargetConfig instance_;
        return instance_;
    }

    /// Constructor queries the environment variable or sets to default value.
    TargetConfig()
    {
        target_ = slate::Target::HostTask;
        const char* str = std::getenv( "SLATE_LAPACK_TARGET" );
        if (str) {
            std::string str_ = str;
            std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );
            if (str_ == "devices")
                target_ = slate::Target::Devices;
            else if (str_ == "hosttask")
                target_ = slate::Target::HostTask;
            else if (str_ == "hostnest")
                target_ = slate::Target::HostNest;
            else if (str_ == "hostbatch")
                target_ = slate::Target::HostBatch;
            else
                slate_error( std::string( "Invalid target: " ) + str );
        }
    }

    // Prevent copy construction and copy assignment.
    TargetConfig( const TargetConfig& orig ) = delete;
    TargetConfig& operator= ( const TargetConfig& orig ) = delete;

    //----------------------------------------
    // Data
    slate::Target target_;
};

//==============================================================================
/// Initialize panel threads setting from environment variable.
/// Uses thread-safe Scott Meyers Singleton.
class PanelThreadsConfig
{
public:
    /// @return number of panel threads to use.
    static int value()
    {
        return instance().panel_threads_;
    }

    /// Set number of panel threads to use.
    static void value( int panel_threads )
    {
        instance().panel_threads_ = panel_threads;
    }

private:
    /// On first call, creates the singleton instance, which queries the
    /// environment variable.
    /// @return singleton instance.
    static PanelThreadsConfig& instance()
    {
        static PanelThreadsConfig instance_;
        return instance_;
    }

    /// Constructor queries the environment variable or sets to default value.
    PanelThreadsConfig()
    {
        panel_threads_ = blas::max( omp_get_max_threads()/2, 1 );
        const char* str = std::getenv( "SLATE_LAPACK_PANELTHREADS" );
        if (str) {
            panel_threads_ = blas::max( strtol( str, NULL, 0 ), 1 );
        }
    }

    // Prevent copy construction and copy assignment.
    PanelThreadsConfig( const PanelThreadsConfig& orig ) = delete;
    PanelThreadsConfig& operator= ( const PanelThreadsConfig& orig ) = delete;

    //----------------------------------------
    // Data
    int panel_threads_;
};

//==============================================================================
/// Initialize ib setting from environment variable.
/// Uses thread-safe Scott Meyers Singleton.
class IBConfig
{
public:
    /// @return inner blocking to use.
    static int64_t value()
    {
        return instance().ib_;
    }

    /// Set inner blocking to use.
    static void value( int64_t ib )
    {
        instance().ib_ = ib;
    }

private:
    /// On first call, creates the singleton instance, which queries the
    /// environment variable.
    /// @return singleton instance.
    static IBConfig& instance()
    {
        static IBConfig instance_;
        return instance_;
    }

    /// Constructor queries the environment variable or sets to default value.
    IBConfig()
    {
        ib_ = 16;
        const char* str = std::getenv( "SLATE_LAPACK_IB" );
        if (str) {
            ib_ = blas::max( strtol( str, NULL, 0 ), 1 );
        }
    }

    // Prevent copy construction and copy assignment.
    IBConfig( const IBConfig& orig ) = delete;
    IBConfig& operator= ( const IBConfig& orig ) = delete;

    //----------------------------------------
    // Data
    int64_t ib_;
};

//==============================================================================
/// Initialize verbose setting from environment variable.
/// Uses thread-safe Scott Meyers Singleton.
class VerboseConfig
{
public:
    /// @return verbose flag to use.
    static int value()
    {
        return instance().verbose_;
    }

    /// Set verbose flag to use.
    static void value( int verbose )
    {
        instance().verbose_ = verbose;
    }

private:
    /// On first call, creates the singleton instance, which queries the
    /// environment variable.
    /// @return singleton instance.
    static VerboseConfig& instance()
    {
        static VerboseConfig instance_;
        return instance_;
    }

    /// Constructor queries the environment variable or sets to default value.
    VerboseConfig()
    {
        verbose_ = 0;
        const char* str = std::getenv( "SLATE_LAPACK_VERBOSE" );
        if (str) {
            verbose_ = strtol( str, NULL, 0 );
        }
    }

    // Prevent copy construction and copy assignment.
    VerboseConfig( const VerboseConfig& orig ) = delete;
    VerboseConfig& operator= ( const VerboseConfig& orig ) = delete;

    //----------------------------------------
    // Data
    int verbose_;
};

//==============================================================================
/// Initialize lookahead setting from environment variable.
/// Uses thread-safe Scott Meyers Singleton.
class LookaheadConfig
{
public:
    /// @return lookahead to use.
    static int64_t value()
    {
        return instance().lookahead_;
    }

    /// Set lookahead to use.
    static void value( int64_t lookahead )
    {
        instance().lookahead_ = lookahead;
    }

private:
    /// On first call, creates the singleton instance, which queries the
    /// environment variable.
    /// @return singleton instance.
    static LookaheadConfig& instance()
    {
        static LookaheadConfig instance_;
        return instance_;
    }

    /// Constructor queries the environment variable or sets to default value.
    LookaheadConfig()
    {
        lookahead_ = 1;
        const char* str = std::getenv( "SLATE_LAPACK_LOOKAHEAD" );
        if (str) {
            lookahead_ = blas::max( strtol( str, NULL, 0 ), 1 );
        }
    }

    // Prevent copy construction and copy assignment.
    LookaheadConfig( const LookaheadConfig& orig ) = delete;
    LookaheadConfig& operator= ( const LookaheadConfig& orig ) = delete;

    //----------------------------------------
    // Data
    int64_t lookahead_;
};

//==============================================================================
/// Initialize nb setting from environment variable.
/// Uses thread-safe Scott Meyers Singleton.
class NBConfig
{
public:
    /// @return inner blocking to use.
    static int64_t value()
    {
        return instance().nb_;
    }

    /// Set inner blocking to use.
    static void value( int64_t nb )
    {
        instance().nb_ = nb;
    }

private:
    /// On first call, creates the singleton instance, which queries the
    /// environment variable.
    /// @return singleton instance.
    static NBConfig& instance()
    {
        static NBConfig instance_;
        return instance_;
    }

    /// Constructor queries the environment variable or sets to default value.
    NBConfig()
    {
        nb_ = 384;
        const char* str = std::getenv( "SLATE_LAPACK_NB" );
        if (str) {
            nb_ = blas::max( strtol( str, NULL, 0 ), 1 );
        }
    }

    // Prevent copy construction and copy assignment.
    NBConfig( const NBConfig& orig ) = delete;
    NBConfig& operator= ( const NBConfig& orig ) = delete;

    //----------------------------------------
    // Data
    int64_t nb_;
};

} // namespace lapack_api
} // namespace slate

#endif // SLATE_LAPACK_API_COMMON_HH
