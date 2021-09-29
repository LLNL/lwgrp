##
# Prologue.
##
AC_INIT([Light-weight Group Library],[1.0.4],[],[lwgrp],[https://github.com/hpc/lwgrp])
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config/config.h])
AC_CONFIG_SRCDIR([src/lwgrp.c])

#AC_CANONICAL_SYSTEM

##
# Automake support.
##
AM_INIT_AUTOMAKE([foreign])

#AM_MAINTAINER_MODE

##
# Checks for programs.
##
AC_PROG_CC
AM_PROG_CC_C_O
AC_PROG_CXX
LX_FIND_MPI

  # don't use MPI_ANY_SOURCE unless enabled
  AC_ARG_ENABLE(
    [mpianysource],
    AS_HELP_STRING(--enable-mpianysource,Specify whether to use MPI_ANY_SOURCE),
    AC_DEFINE([LWGRP_USE_ANYSOURCE], [1], [Define to 1 if liblwgrp should use MPI_ANY_SOURCE])
  )

##
# enable libtool
##
AC_PROG_LIBTOOL

##
# Extra packages
##

##
# Checks for header files.
##
AC_HEADER_STDC

##
# Checks for typedefs, structures, and compiler characteristics.
##
AC_C_BIGENDIAN

##
# Checks for library functions.
##

##
# Epilogue.
##
AC_CONFIG_FILES([ \
  Makefile \
  src/Makefile \
  liblwgrp.pc \
])
AC_OUTPUT

