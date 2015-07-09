#pragma once

#if defined(EIGEN_RUNTIME_NO_MALLOC)
  #define EQP_CHECK_MALLOC(x) Eigen::internal::set_is_malloc_allowed(!x)
#else
  #define EQP_CHECK_MALLOC(x)
#endif
