var documenterSearchIndex = {"docs":
[{"location":"Testing/test/","page":"-","title":"-","text":"","category":"page"},{"location":"Testing/test/","page":"-","title":"-","text":"author: \"Chris Rackauckas\" title: \"Test\" –-","category":"page"},{"location":"Testing/test/","page":"-","title":"-","text":"This is a test of the builder system.  It often gets bumped manually.","category":"page"},{"location":"Testing/test/#Appendix","page":"-","title":"Appendix","text":"","category":"section"},{"location":"Testing/test/","page":"-","title":"-","text":"These tutorials are a part of the SciMLTutorials.jl repository, found at: https://github.com/SciML/SciMLTutorials.jl. For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization https://sciml.ai.","category":"page"},{"location":"Testing/test/","page":"-","title":"-","text":"To locally run this tutorial, do the following commands:","category":"page"},{"location":"Testing/test/","page":"-","title":"-","text":"using SciMLTutorials\nSciMLTutorials.weave_file(\"Testing\",\"test.jmd\")","category":"page"},{"location":"Testing/test/","page":"-","title":"-","text":"Computer Information:","category":"page"},{"location":"Testing/test/","page":"-","title":"-","text":"Julia Version 1.6.5\nCommit 9058264a69 (2021-12-19 12:30 UTC)\nPlatform Info:\n  OS: Linux (x86_64-pc-linux-gnu)\n  CPU: AMD EPYC 7502 32-Core Processor\n  WORD_SIZE: 64\n  LIBM: libopenlibm\n  LLVM: libLLVM-11.0.1 (ORCJIT, znver2)\nEnvironment:\n  JULIA_CPU_THREADS = 16\n  BUILDKITE_PLUGIN_JULIA_CACHE_DIR = /cache/julia-buildkite-plugin\n  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/a6029d3a-f78b-41ea-bc97-28aa57c6c6ea\n","category":"page"},{"location":"Testing/test/","page":"-","title":"-","text":"Package Information:","category":"page"},{"location":"Testing/test/","page":"-","title":"-","text":"      Status `/cache/build/default-amdci4-1/julialang/scimltutorials-dot-jl/tutorials/Testing/Project.toml`\n  [30cb0354] SciMLTutorials v0.9.0","category":"page"},{"location":"Testing/test/","page":"-","title":"-","text":"And the full manifest:","category":"page"},{"location":"Testing/test/","page":"-","title":"-","text":"      Status `/cache/build/default-amdci4-1/julialang/scimltutorials-dot-jl/tutorials/Testing/Manifest.toml`\n  [79e6a3ab] Adapt v3.3.0\n  [35d6a980] ColorSchemes v3.12.1\n  [3da002f7] ColorTypes v0.11.0\n  [5ae59095] Colors v0.12.8\n  [34da2185] Compat v3.30.0\n  [8f4d0f93] Conda v1.5.2\n  [d38c429a] Contour v0.5.7\n  [9a962f9c] DataAPI v1.6.0\n  [864edb3b] DataStructures v0.18.9\n  [e2d170a0] DataValueInterfaces v1.0.0\n  [ffbed154] DocStringExtensions v0.8.4\n  [c87230d0] FFMPEG v0.4.0\n  [53c48c17] FixedPointNumbers v0.8.4\n  [59287772] Formatting v0.4.2\n  [28b8d3ca] GR v0.57.4\n  [5c1252a2] GeometryBasics v0.3.12\n  [42e2da0e] Grisu v1.0.2\n  [cd3eb016] HTTP v0.9.9\n  [eafb193a] Highlights v0.4.5\n  [7073ff75] IJulia v1.23.2\n  [83e8ac13] IniFile v0.5.0\n  [c8e1da08] IterTools v1.3.0\n  [82899510] IteratorInterfaceExtensions v1.0.0\n  [692b3bcd] JLLWrappers v1.3.0\n  [682c06a0] JSON v0.21.1\n  [b964fa9f] LaTeXStrings v1.2.1\n  [23fbe1c1] Latexify v0.15.5\n  [1914dd2f] MacroTools v0.5.6\n  [739be429] MbedTLS v1.0.3\n  [442fdcdd] Measures v0.3.1\n  [e1d29d7a] Missings v1.0.0\n  [ffc61752] Mustache v1.0.10\n  [77ba4419] NaNMath v0.3.5\n  [bac558e1] OrderedCollections v1.4.1\n  [69de0a69] Parsers v1.1.0\n  [ccf2f8ad] PlotThemes v2.0.1\n  [995b91a9] PlotUtils v1.0.10\n  [91a5bcdd] Plots v1.15.2\n  [21216c6a] Preferences v1.2.2\n  [3cdcf5f2] RecipesBase v1.1.1\n  [01d81517] RecipesPipeline v0.3.2\n  [189a3867] Reexport v1.0.0\n  [ae029012] Requires v1.1.3\n  [30cb0354] SciMLTutorials v0.9.0\n  [6c6a2e73] Scratch v1.0.3\n  [992d4aef] Showoff v1.0.3\n  [b85f4697] SoftGlobalScope v1.1.0\n  [a2af1166] SortingAlgorithms v1.0.0\n  [90137ffa] StaticArrays v1.2.0\n  [82ae8749] StatsAPI v1.0.0\n  [2913bbd2] StatsBase v0.33.8\n  [09ab397b] StructArrays v0.5.1\n  [3783bdb8] TableTraits v1.0.1\n  [bd369af6] Tables v1.4.2\n  [5c2747f8] URIs v1.3.0\n  [81def892] VersionParsing v1.2.0\n  [44d3d7a6] Weave v0.10.8\n  [ddb6d928] YAML v0.4.6\n  [c2297ded] ZMQ v1.2.1\n  [6e34b625] Bzip2_jll v1.0.6+5\n  [83423d85] Cairo_jll v1.16.0+6\n  [5ae413db] EarCut_jll v2.1.5+1\n  [2e619515] Expat_jll v2.2.10+0\n  [b22a6f82] FFMPEG_jll v4.3.1+4\n  [a3f928ae] Fontconfig_jll v2.13.1+14\n  [d7e528f0] FreeType2_jll v2.10.1+5\n  [559328eb] FriBidi_jll v1.0.5+6\n  [0656b61e] GLFW_jll v3.3.4+0\n  [d2c73de3] GR_jll v0.57.2+0\n  [78b55507] Gettext_jll v0.21.0+0\n  [7746bdde] Glib_jll v2.68.1+0\n  [aacddb02] JpegTurbo_jll v2.0.1+3\n  [c1c5ebd0] LAME_jll v3.100.0+3\n  [dd4b983a] LZO_jll v2.10.1+0\n  [dd192d2f] LibVPX_jll v1.9.0+1\n  [e9f186c6] Libffi_jll v3.2.2+0\n  [d4300ac3] Libgcrypt_jll v1.8.7+0\n  [7e76a0d4] Libglvnd_jll v1.3.0+3\n  [7add5ba3] Libgpg_error_jll v1.42.0+0\n  [94ce4f54] Libiconv_jll v1.16.1+0\n  [4b2f31a3] Libmount_jll v2.35.0+0\n  [89763e89] Libtiff_jll v4.1.0+2\n  [38a345b3] Libuuid_jll v2.36.0+0\n  [e7412a2a] Ogg_jll v1.3.4+2\n  [458c3c95] OpenSSL_jll v1.1.1+6\n  [91d4177d] Opus_jll v1.3.1+3\n  [2f80f16e] PCRE_jll v8.44.0+0\n  [30392449] Pixman_jll v0.40.1+0\n  [ea2cea3b] Qt5Base_jll v5.15.2+0\n  [a2964d1f] Wayland_jll v1.17.0+4\n  [2381bf8a] Wayland_protocols_jll v1.18.0+4\n  [02c8fc9c] XML2_jll v2.9.12+0\n  [aed1982a] XSLT_jll v1.1.34+0\n  [4f6342f7] Xorg_libX11_jll v1.6.9+4\n  [0c0b7dd1] Xorg_libXau_jll v1.0.9+4\n  [935fb764] Xorg_libXcursor_jll v1.2.0+4\n  [a3789734] Xorg_libXdmcp_jll v1.1.3+4\n  [1082639a] Xorg_libXext_jll v1.3.4+4\n  [d091e8ba] Xorg_libXfixes_jll v5.0.3+4\n  [a51aa0fd] Xorg_libXi_jll v1.7.10+4\n  [d1454406] Xorg_libXinerama_jll v1.1.4+4\n  [ec84b674] Xorg_libXrandr_jll v1.5.2+4\n  [ea2f1a96] Xorg_libXrender_jll v0.9.10+4\n  [14d82f49] Xorg_libpthread_stubs_jll v0.1.0+3\n  [c7cfdc94] Xorg_libxcb_jll v1.13.0+3\n  [cc61e674] Xorg_libxkbfile_jll v1.1.0+4\n  [12413925] Xorg_xcb_util_image_jll v0.4.0+1\n  [2def613f] Xorg_xcb_util_jll v0.4.0+1\n  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.0+1\n  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.9+1\n  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.1+1\n  [35661453] Xorg_xkbcomp_jll v1.4.2+4\n  [33bec58e] Xorg_xkeyboard_config_jll v2.27.0+4\n  [c5fb5394] Xorg_xtrans_jll v1.4.0+3\n  [8f1865be] ZeroMQ_jll v4.3.2+6\n  [3161d3a3] Zstd_jll v1.5.0+0\n  [0ac62f75] libass_jll v0.14.0+4\n  [f638f0a6] libfdk_aac_jll v0.1.6+4\n  [b53b4c65] libpng_jll v1.6.38+0\n  [a9144af2] libsodium_jll v1.0.20+0\n  [f27f6e37] libvorbis_jll v1.3.6+6\n  [1270edf5] x264_jll v2020.7.14+2\n  [dfaa095f] x265_jll v3.0.0+3\n  [d8fb68d0] xkbcommon_jll v0.9.1+5\n  [0dad84c5] ArgTools\n  [56f22d72] Artifacts\n  [2a0f44e3] Base64\n  [ade2ca70] Dates\n  [8bb1440f] DelimitedFiles\n  [8ba89e20] Distributed\n  [f43a241f] Downloads\n  [7b1f6079] FileWatching\n  [b77e0a4c] InteractiveUtils\n  [b27032c2] LibCURL\n  [76f85450] LibGit2\n  [8f399da3] Libdl\n  [37e2e46d] LinearAlgebra\n  [56ddb016] Logging\n  [d6f4376e] Markdown\n  [a63ad114] Mmap\n  [ca575930] NetworkOptions\n  [44cfe95a] Pkg\n  [de0858da] Printf\n  [3fa0cd96] REPL\n  [9a3f8284] Random\n  [ea8e919c] SHA\n  [9e88b42a] Serialization\n  [1a1011a3] SharedArrays\n  [6462fe0b] Sockets\n  [2f01184e] SparseArrays\n  [10745b16] Statistics\n  [fa267f1f] TOML\n  [a4e569a6] Tar\n  [8dfed614] Test\n  [cf7118a7] UUIDs\n  [4ec0a83e] Unicode\n  [e66e0078] CompilerSupportLibraries_jll\n  [deac9b47] LibCURL_jll\n  [29816b5a] LibSSH2_jll\n  [c8ffd9c3] MbedTLS_jll\n  [14a3606d] MozillaCACerts_jll\n  [83775a58] Zlib_jll\n  [8e850ede] nghttp2_jll\n  [3f19e933] p7zip_jll","category":"page"},{"location":"test/","page":"-","title":"-","text":"","category":"page"},{"location":"test/","page":"-","title":"-","text":"author: \"Chris Rackauckas\" title: \"Test\" –-","category":"page"},{"location":"test/","page":"-","title":"-","text":"This is a test of the builder system.","category":"page"},{"location":"test/#Appendix","page":"-","title":"Appendix","text":"","category":"section"},{"location":"test/","page":"-","title":"-","text":"This tutorial is part of the SciMLTutorials.jl repository, found at: https://github.com/SciML/SciMLTutorials.jl.  For more information on doing scientific machine learning (SciML) with open source software, check out https://sciml.ai/.","category":"page"},{"location":"test/","page":"-","title":"-","text":"To locally run this tutorial, do the following commands:","category":"page"},{"location":"test/","page":"-","title":"-","text":"using SciMLTutorials\nSciMLTutorials.weave_file(\".\",\"Testing/test.jmd\")","category":"page"},{"location":"test/","page":"-","title":"-","text":"Computer Information:","category":"page"},{"location":"test/","page":"-","title":"-","text":"Julia Version 1.6.1\nCommit 6aaedecc44 (2021-04-23 05:59 UTC)\nPlatform Info:\n  OS: macOS (x86_64-apple-darwin18.7.0)\n  CPU: Intel(R) Core(TM) i5-5350U CPU @ 1.80GHz\n  WORD_SIZE: 64\n  LIBM: libopenlibm\n  LLVM: libLLVM-11.0.1 (ORCJIT, broadwell)\nEnvironment:\n  JULIA_LOAD_PATH = @:/var/folders/lv/kg1z6t3s4wdf1sss2p42dr200000gn/T/jl_NMW2d4\n","category":"page"},{"location":"test/","page":"-","title":"-","text":"      Status `~/.julia/dev/SciMLTutorials/tutorials/Testing/Project.toml`\n  [30cb0354] SciMLTutorials v0.8.0","category":"page"},{"location":"test/","page":"-","title":"-","text":"And the full manifest:","category":"page"},{"location":"test/","page":"-","title":"-","text":"      Status `~/.julia/dev/SciMLTutorials/tutorials/Testing/Manifest.toml`\n  [621f4979] AbstractFFTs v1.0.1\n  [79e6a3ab] Adapt v3.3.0\n  [d360d2e6] ChainRulesCore v0.9.44\n  [35d6a980] ColorSchemes v3.12.1\n  [3da002f7] ColorTypes v0.11.0\n  [5ae59095] Colors v0.12.8\n  [34da2185] Compat v3.30.0\n  [8f4d0f93] Conda v1.5.2\n  [d38c429a] Contour v0.5.7\n  [717857b8] DSP v0.6.10\n  [9a962f9c] DataAPI v1.6.0\n  [864edb3b] DataStructures v0.18.9\n  [e2d170a0] DataValueInterfaces v1.0.0\n  [ffbed154] DocStringExtensions v0.8.4\n  [e2ba6199] ExprTools v0.1.3\n  [8f5d6c58] EzXML v1.1.0\n  [c87230d0] FFMPEG v0.4.0\n  [7a1cc6ca] FFTW v1.4.1\n  [53c48c17] FixedPointNumbers v0.8.4\n  [59287772] Formatting v0.4.2\n  [28b8d3ca] GR v0.57.4\n  [5c1252a2] GeometryBasics v0.3.12\n  [42e2da0e] Grisu v1.0.2\n  [cd3eb016] HTTP v0.9.8\n  [eafb193a] Highlights v0.4.5\n  [7073ff75] IJulia v1.23.2\n  [83e8ac13] IniFile v0.5.0\n  [d8418881] Intervals v1.5.0\n  [c8e1da08] IterTools v1.3.0\n  [82899510] IteratorInterfaceExtensions v1.0.0\n  [692b3bcd] JLLWrappers v1.3.0\n  [682c06a0] JSON v0.21.1\n  [b964fa9f] LaTeXStrings v1.2.1\n  [23fbe1c1] Latexify v0.15.5\n  [2ab3a3ac] LogExpFunctions v0.2.4\n  [1914dd2f] MacroTools v0.5.6\n  [739be429] MbedTLS v1.0.3\n  [442fdcdd] Measures v0.3.1\n  [e1d29d7a] Missings v1.0.0\n  [78c3b35d] Mocking v0.7.1\n  [ffc61752] Mustache v1.0.10\n  [77ba4419] NaNMath v0.3.5\n  [6fe1bfb0] OffsetArrays v1.8.0\n  [bac558e1] OrderedCollections v1.4.1\n  [69de0a69] Parsers v1.1.0\n  [ccf2f8ad] PlotThemes v2.0.1\n  [995b91a9] PlotUtils v1.0.10\n  [91a5bcdd] Plots v1.15.2\n  [f27b6e38] Polynomials v1.2.1\n  [21216c6a] Preferences v1.2.2\n  [3cdcf5f2] RecipesBase v1.1.1\n  [01d81517] RecipesPipeline v0.3.2\n  [189a3867] Reexport v1.0.0\n  [ae029012] Requires v1.1.3\n  [30cb0354] SciMLTutorials v0.8.0\n  [6c6a2e73] Scratch v1.0.3\n  [992d4aef] Showoff v1.0.3\n  [b85f4697] SoftGlobalScope v1.1.0\n  [a2af1166] SortingAlgorithms v1.0.0\n  [276daf66] SpecialFunctions v1.4.1\n  [90137ffa] StaticArrays v1.2.0\n  [82ae8749] StatsAPI v1.0.0\n  [2913bbd2] StatsBase v0.33.8\n  [09ab397b] StructArrays v0.5.1\n  [3783bdb8] TableTraits v1.0.1\n  [bd369af6] Tables v1.4.2\n  [f269a46b] TimeZones v1.5.5\n  [5c2747f8] URIs v1.3.0\n  [81def892] VersionParsing v1.2.0\n  [44d3d7a6] Weave v0.10.8\n  [ddb6d928] YAML v0.4.6\n  [c2297ded] ZMQ v1.2.1\n  [6e34b625] Bzip2_jll v1.0.6+5\n  [83423d85] Cairo_jll v1.16.0+6\n  [5ae413db] EarCut_jll v2.1.5+1\n  [2e619515] Expat_jll v2.2.10+0\n  [b22a6f82] FFMPEG_jll v4.3.1+4\n  [f5851436] FFTW_jll v3.3.9+7\n  [a3f928ae] Fontconfig_jll v2.13.1+14\n  [d7e528f0] FreeType2_jll v2.10.1+5\n  [559328eb] FriBidi_jll v1.0.5+6\n  [0656b61e] GLFW_jll v3.3.4+0\n  [d2c73de3] GR_jll v0.57.2+0\n  [78b55507] Gettext_jll v0.21.0+0\n  [7746bdde] Glib_jll v2.68.1+0\n  [1d5cc7b8] IntelOpenMP_jll v2018.0.3+2\n  [aacddb02] JpegTurbo_jll v2.0.1+3\n  [c1c5ebd0] LAME_jll v3.100.0+3\n  [dd4b983a] LZO_jll v2.10.0+3\n  [dd192d2f] LibVPX_jll v1.9.0+1\n  [e9f186c6] Libffi_jll v3.2.2+0\n  [d4300ac3] Libgcrypt_jll v1.8.5+4\n  [7e76a0d4] Libglvnd_jll v1.3.0+3\n  [7add5ba3] Libgpg_error_jll v1.36.0+3\n  [94ce4f54] Libiconv_jll v1.16.1+0\n  [4b2f31a3] Libmount_jll v2.35.0+0\n  [89763e89] Libtiff_jll v4.1.0+2\n  [38a345b3] Libuuid_jll v2.36.0+0\n  [856f044c] MKL_jll v2021.1.1+1\n  [e7412a2a] Ogg_jll v1.3.4+2\n  [458c3c95] OpenSSL_jll v1.1.1+6\n  [efe28fd5] OpenSpecFun_jll v0.5.4+0\n  [91d4177d] Opus_jll v1.3.1+3\n  [2f80f16e] PCRE_jll v8.44.0+0\n  [30392449] Pixman_jll v0.40.0+0\n  [ea2cea3b] Qt5Base_jll v5.15.2+0\n  [a2964d1f] Wayland_jll v1.17.0+4\n  [2381bf8a] Wayland_protocols_jll v1.18.0+4\n  [02c8fc9c] XML2_jll v2.9.12+0\n  [aed1982a] XSLT_jll v1.1.33+4\n  [4f6342f7] Xorg_libX11_jll v1.6.9+4\n  [0c0b7dd1] Xorg_libXau_jll v1.0.9+4\n  [935fb764] Xorg_libXcursor_jll v1.2.0+4\n  [a3789734] Xorg_libXdmcp_jll v1.1.3+4\n  [1082639a] Xorg_libXext_jll v1.3.4+4\n  [d091e8ba] Xorg_libXfixes_jll v5.0.3+4\n  [a51aa0fd] Xorg_libXi_jll v1.7.10+4\n  [d1454406] Xorg_libXinerama_jll v1.1.4+4\n  [ec84b674] Xorg_libXrandr_jll v1.5.2+4\n  [ea2f1a96] Xorg_libXrender_jll v0.9.10+4\n  [14d82f49] Xorg_libpthread_stubs_jll v0.1.0+3\n  [c7cfdc94] Xorg_libxcb_jll v1.13.0+3\n  [cc61e674] Xorg_libxkbfile_jll v1.1.0+4\n  [12413925] Xorg_xcb_util_image_jll v0.4.0+1\n  [2def613f] Xorg_xcb_util_jll v0.4.0+1\n  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.0+1\n  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.9+1\n  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.1+1\n  [35661453] Xorg_xkbcomp_jll v1.4.2+4\n  [33bec58e] Xorg_xkeyboard_config_jll v2.27.0+4\n  [c5fb5394] Xorg_xtrans_jll v1.4.0+3\n  [8f1865be] ZeroMQ_jll v4.3.2+6\n  [3161d3a3] Zstd_jll v1.5.0+0\n  [0ac62f75] libass_jll v0.14.0+4\n  [f638f0a6] libfdk_aac_jll v0.1.6+4\n  [b53b4c65] libpng_jll v1.6.37+6\n  [a9144af2] libsodium_jll v1.0.20+0\n  [f27f6e37] libvorbis_jll v1.3.6+6\n  [1270edf5] x264_jll v2020.7.14+2\n  [dfaa095f] x265_jll v3.0.0+3\n  [d8fb68d0] xkbcommon_jll v0.9.1+5\n  [0dad84c5] ArgTools\n  [56f22d72] Artifacts\n  [2a0f44e3] Base64\n  [ade2ca70] Dates\n  [8bb1440f] DelimitedFiles\n  [8ba89e20] Distributed\n  [f43a241f] Downloads\n  [7b1f6079] FileWatching\n  [b77e0a4c] InteractiveUtils\n  [4af54fe1] LazyArtifacts\n  [b27032c2] LibCURL\n  [76f85450] LibGit2\n  [8f399da3] Libdl\n  [37e2e46d] LinearAlgebra\n  [56ddb016] Logging\n  [d6f4376e] Markdown\n  [a63ad114] Mmap\n  [ca575930] NetworkOptions\n  [44cfe95a] Pkg\n  [de0858da] Printf\n  [3fa0cd96] REPL\n  [9a3f8284] Random\n  [ea8e919c] SHA\n  [9e88b42a] Serialization\n  [1a1011a3] SharedArrays\n  [6462fe0b] Sockets\n  [2f01184e] SparseArrays\n  [10745b16] Statistics\n  [fa267f1f] TOML\n  [a4e569a6] Tar\n  [8dfed614] Test\n  [cf7118a7] UUIDs\n  [4ec0a83e] Unicode\n  [e66e0078] CompilerSupportLibraries_jll\n  [deac9b47] LibCURL_jll\n  [29816b5a] LibSSH2_jll\n  [c8ffd9c3] MbedTLS_jll\n  [14a3606d] MozillaCACerts_jll\n  [83775a58] Zlib_jll\n  [8e850ede] nghttp2_jll\n  [3f19e933] p7zip_jll","category":"page"},{"location":"#SciMLTutorials.jl:-Tutorials-for-Scientific-Machine-Learning-and-Differential-Equations","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning and Differential Equations","text":"","category":"section"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"(Image: Join the chat at https://julialang.zulipchat.com #sciml-bridged) (Image: Global Docs)","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"(Image: Build status)","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"(Image: ColPrac: Contributor's Guide on Collaborative Practices for Community Packages) (Image: SciML Code Style)","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"SciMLTutorials.jl holds PDFs, webpages, and interactive Jupyter notebooks showing how to utilize the software in the SciML Scientific Machine Learning ecosystem. This set of tutorials was made to complement the documentation and the devdocs by providing practical examples of the concepts. For more details, please consult the docs.","category":"page"},{"location":"#Note:-this-library-has-been-deprecated-and-its-tutorials-have-been-moved-to-the-repos-of-the-respective-packages.-It-may-be-revived-in-the-future-if-there-is-a-need-for-longer-form-tutorials!","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"Note: this library has been deprecated and its tutorials have been moved to the repos of the respective packages. It may be revived in the future if there is a need for longer-form tutorials!","text":"","category":"section"},{"location":"#Results","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"Results","text":"","category":"section"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"To view the SciML Tutorials, go to tutorials.sciml.ai. By default, this will lead to the latest tagged version of the tutorials. To see the in-development version of the tutorials, go to https://tutorials.sciml.ai/dev/.","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"Static outputs in pdf, markdown, and html reside in SciMLTutorialsOutput.","category":"page"},{"location":"#Video-Tutorial","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"Video Tutorial","text":"","category":"section"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"(Image: Video Tutorial)","category":"page"},{"location":"#Interactive-Notebooks","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"Interactive Notebooks","text":"","category":"section"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"To generate the interactive notebooks, first install the SciMLTutorials, instantiate the environment, and then run SciMLTutorials.open_notebooks(). This looks as follows:","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"]add SciMLTutorials#master\n]activate SciMLTutorials\n]instantiate\nusing SciMLTutorials\nSciMLTutorials.open_notebooks()","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"The tutorials will be generated at your pwd() in a folder called generated_notebooks.","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"Note that when running the tutorials, the packages are not automatically added. Thus you will need to add the packages manually or use the internal Project/Manifest tomls to instantiate the correct packages. This can be done by activating the folder of the tutorials. For example,","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"using Pkg\nPkg.activate(joinpath(pkgdir(SciMLTutorials),\"tutorials\",\"models\"))\nPkg.instantiate()","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"will add all of the packages required to run any tutorial in the models folder.","category":"page"},{"location":"#Contributing","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"Contributing","text":"","category":"section"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"All of the files are generated from the Weave.jl files in the tutorials folder. The generation process runs automatically, and thus one does not necessarily need to test the Weave process locally. Instead, simply open a PR that adds/updates a file in the \"tutorials\" folder and the PR will generate the tutorial on demand. Its artifacts can then be inspected in the Buildkite as described below before merging. Note that it will use the Project.toml and Manifest.toml of the subfolder, so any changes to dependencies requires that those are updated.","category":"page"},{"location":"#Reporting-Bugs-and-Issues","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"Reporting Bugs and Issues","text":"","category":"section"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"Report any bugs or issues at the SciMLTutorials repository.","category":"page"},{"location":"#Inspecting-Tutorial-Results","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"Inspecting Tutorial Results","text":"","category":"section"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"To see tutorial results before merging, click into the BuildKite, click onto Artifacts, and then investigate the trained results.","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"(Image: )","category":"page"},{"location":"#Manually-Generating-Files","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"Manually Generating Files","text":"","category":"section"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"To run the generation process, do for example:","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"]activate SciMLTutorials # Get all of the packages\nusing SciMLTutorials\nSciMLTutorials.weave_file(joinpath(pkgdir(SciMLTutorials),\"tutorials\",\"models\"),\"01-classical_physics.jmd\")","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"To generate all of the files in a folder, for example, run:","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"SciMLTutorials.weave_folder(joinpath(pkgdir(SciMLTutorials),\"tutorials\",\"models\"))","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"To generate all of the notebooks, do:","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"SciMLTutorials.weave_all()","category":"page"},{"location":"","page":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","title":"SciMLTutorials.jl: Tutorials for Scientific Machine Learning (SciML) and Equation Solvers","text":"Each of the tuturials displays the computer characteristics at the bottom of the benchmark.","category":"page"}]
}
