/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// CuTe DSL prebuilt CUDA 12.x headers use cudaLibrary_t/cudaLibraryUnload.
// Some embedded CUDA 12.0-12.7 runtime headers do not declare those runtime
// library-management APIs even though the driver API provides CUlibrary. The
// C shim supplies the link-time symbols; these declarations make the generated
// headers compile.
#if defined(CUDART_VERSION) && CUDART_VERSION < 12080
typedef CUlibrary cudaLibrary_t;

extern "C" cudaError_t cudaLibraryUnload(cudaLibrary_t library);
#endif
