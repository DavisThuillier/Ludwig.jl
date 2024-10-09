# Installation

Ludwig.jl is not yet registered on the [general Julia package registry](https://github.com/JuliaRegistries/General). 
To run the latest version of Ludwig.jl, clone the repository. In the root folder of the project, 
```
julia> ]
(@v1.8) pkg> activate .
(Ludwig) pkg> instantiate
```
To run scripts using Ludwig, you must check out Ludwig for development in your scripting environment.
```
julia> ]
(@v1.8) pkg> activate PATH_TO_SCRIPT_DIR
(SCRIPT_DIR) pkg> dev PATH_TO_LUDWIG_ROOT
(SCRIPT_DIR) pkg> instantiate
```
Functionality provided by Ludwig can now be accessed with `using Ludwig`.