# ==================== test_sixs.py ====================
"""
测试6S模型基本功能
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Py6S import *
import numpy as np


def test_sixs_basic():
    """测试6S基本功能"""
    print("=" * 60)
    print("测试6S基本功能")
    print("=" * 60)

    # 创建6S实例
    s = SixS()

    # 设置波长
    s.wavelength = Wavelength(0.64)

    # 设置大气廓线
    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)

    # 设置气溶胶
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    s.aot550 = 0.2

    # 设置几何参数
    s.geometry = Geometry.User()
    s.geometry.solar_z = 30.0
    s.geometry.solar_a = 0.0
    s.geometry.view_z = 0.0
    s.geometry.view_a = 0.0

    # 设置海拔
    s.altitudes = Altitudes()
    s.altitudes.set_target_custom_altitude(0.0)
    s.altitudes.set_sensor_satellite_level()

    try:
        # 测试1: 正向模拟
        print("\n1. 测试正向模拟")
        s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.2)
        s.atmos_corr = AtmosCorr.NoAtmosCorr()

        s.run()

        print(f"TOA反射率: {s.outputs.values['apparent_reflectance']:.6f}")
        print(f"地表反射率: {s.outputs.values['pixel_reflectance']:.6f}")

        # 测试2: 反演模拟
        print("\n2. 测试反演模拟")
        rho_toa = s.outputs.values['apparent_reflectance']

        # 创建新的6S实例进行反演
        s_inv = SixS()
        s_inv.wavelength = Wavelength(0.64)
        s_inv.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)
        s_inv.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
        s_inv.aot550 = 0.2

        s_inv.geometry = Geometry.User()
        s_inv.geometry.solar_z = 30.0
        s_inv.geometry.solar_a = 0.0
        s_inv.geometry.view_z = 0.0
        s_inv.geometry.view_a = 0.0

        s_inv.altitudes = Altitudes()
        s_inv.altitudes.set_target_custom_altitude(0.0)
        s_inv.altitudes.set_sensor_satellite_level()

        # 设置大气校正（反演模式）
        s_inv.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(rho_toa)

        s_inv.run()

        print(f"输入TOA反射率: {rho_toa:.6f}")
        print(f"反演地表反射率: {s_inv.outputs.values['pixel_reflectance']:.6f}")
        print(f"原始地表反射率: 0.200000")
        print(f"误差: {s_inv.outputs.values['pixel_reflectance'] - 0.2:.6f}")

        return True

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sixs_with_high_angles():
    """测试高角度条件下的6S模拟"""
    print("\n" + "=" * 60)
    print("测试高角度条件下的6S模拟")
    print("=" * 60)

    try:
        s = SixS()
        s.wavelength = Wavelength(0.64)
        s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)
        s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
        s.aot550 = 0.3

        # 设置高角度
        s.geometry = Geometry.User()
        s.geometry.solar_z = 75.0  # 高太阳天顶角
        s.geometry.solar_a = 0.0
        s.geometry.view_z = 60.0  # 高观测天顶角
        s.geometry.view_a = 180.0  # 后向散射

        s.altitudes = Altitudes()
        s.altitudes.set_target_custom_altitude(0.0)
        s.altitudes.set_sensor_satellite_level()

        # 正向模拟
        s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.2)
        s.atmos_corr = AtmosCorr.NoAtmosCorr()

        s.run()

        rho_toa = s.outputs.values['apparent_reflectance']
        print(f"高角度TOA反射率: {rho_toa:.6f}")

        # 反演
        s_inv = SixS()
        s_inv.wavelength = Wavelength(0.64)
        s_inv.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)
        s_inv.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
        s_inv.aot550 = 0.3
        s_inv.geometry = s.geometry
        s_inv.altitudes = s.altitudes

        s_inv.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(rho_toa)
        s_inv.run()

        print(f"反演地表反射率: {s_inv.outputs.values['pixel_reflectance']:.6f}")
        print(f"误差: {s_inv.outputs.values['pixel_reflectance'] - 0.2:.6f}")

        return True

    except Exception as e:
        print(f"高角度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    success1 = test_sixs_basic()
    success2 = test_sixs_with_high_angles()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("所有测试通过！")
    else:
        print("部分测试失败！")
    print("=" * 60)