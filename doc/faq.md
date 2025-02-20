# Known Issues and FAQ

1. The detector only runs at full speed on the third iteration. This is caused by the need to allocate temporary memory and record kernel firing sequence for cudaGraph. 
2. Why are you not using `texture`? I did some preliminary benchmarking early in development and determined that the overhead is too much.
