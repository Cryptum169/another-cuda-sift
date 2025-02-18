#pragma once

struct CudaSiftConfig {
    int col_width;
    int row_width;
    int numFeatures {5000};
    int numOctaveLayers {3};
    double contrastThreshould {0.04};
    double edgeThreshould = 10;
    double sigma = 1.6;

    // TODO: This is currently broken if set to true.
    bool upscale = false;
};
