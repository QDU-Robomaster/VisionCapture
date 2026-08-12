add_executable(vision_capture_calibration_geometry_test
               calibration_geometry_test.cpp)

target_link_libraries(vision_capture_calibration_geometry_test PRIVATE xr)
target_compile_features(vision_capture_calibration_geometry_test
                        PRIVATE cxx_std_20)

add_test(NAME vision_capture_calibration_geometry_test
         COMMAND vision_capture_calibration_geometry_test)
