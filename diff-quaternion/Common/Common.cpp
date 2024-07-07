#include "Common.h"

namespace DiffRotation
{
    const char * quat_multiply_help = "quaternion (x, y, z, w) multiply."
        "Input: quaternion (x, y, z, w) qa in shape (*, 4), qb in shape (*, 4)"
        "Output: quaternion (x, y, z, w) in shape (*, 4)";

    const char * quat_apply_help = "quaternion apply."
        "Input: quaternion (x, y, z, w) in shape (*, 4), vector in shape (*, 3),"
        "Output: vector in shape (*, 3)";

    const char * quat_inv_help = "quaternion inv. "
         "Input quaternion (x, y, z, w) in shape (*, 4), "
         "Output: quaternion in shape (*, 4)";

    const char * quat_integrate_help = "quaternion integrate."
        "Input: quaternion (x, y, z, w) in shape (*, 4), "
        "angular velocity in shape (*, 3), and delta time"
        "Output: quaternion in shape (*, 4)";

    const char * quat_to_angle_help = "convert quaternion to rotation angle."
        "Input: quaternion in shape (*, 4),"
        "Output: angle in shape (*)";

    const char * quat_to_axis_angle_help = "convert quaternion to axis angle."
        "Input: quaternion in shape (*, 4),"
        "Output: axis angle in shape (*, 3)";

    const char * quat_from_rotvec_help = "get quaternion from axis angle."
        "Input: axis angle in shape (*, 3),"
        "Output: quaternion in shape (*, 4)";

    const char * quat_from_matrix_help = "get quaternion from rotation matrix."
        "Input: matrix in shape (*, 3, 3),"
        "Output: quaternion in shape (*, 4)";
}
