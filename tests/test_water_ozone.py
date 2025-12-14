# ==================== test_water_ozone.py ====================
"""
测试水汽和臭氧设置
"""
from Py6S import *
import numpy as np


def test_water_ozone():
    """测试水汽和臭氧设置"""
    print("=" * 60)
    print("测试Py6S水汽和臭氧设置")
    print("=" * 60)

    # 测试1: 使用预定义大气廓线
    print("\n1. 测试预定义大气廓线")
    s1 = SixS()
    s1.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)
    s1.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    s1.aot550 = 0.2
    s1.wavelength = Wavelength(0.64)

    s1.geometry = Geometry.User()
    s1.geometry.solar_z = 30
    s1.geometry.view_z = 0

    s1.altitudes = Altitudes()
    s1.altitudes.set_target_custom_altitude(0)
    s1.altitudes.set_sensor_satellite_level()

    try:
        s1.run()
        print(f"预定义廓线成功 - TOA反射率: {s1.outputs.apparent_reflectance:.6f}")
    except Exception as e:
        print(f"预定义廓线失败: {e}")

    # 测试2: 使用自定义水汽和臭氧
    print("\n2. 测试自定义水汽和臭氧")
    s2 = SixS()

    # 使用UserWaterAndOzone方法
    water = 2.0  # g/cm²
    ozone = 0.3  # cm-atm
    s2.atmos_profile = AtmosProfile.UserWaterAndOzone(water, ozone)

    s2.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    s2.aot550 = 0.2
    s2.wavelength = Wavelength(0.64)

    s2.geometry = Geometry.User()
    s2.geometry.solar_z = 30
    s2.geometry.view_z = 0

    s2.altitudes = Altitudes()
    s2.altitudes.set_target_custom_altitude(0)
    s2.altitudes.set_sensor_satellite_level()

    try:
        s2.run()
        print(f"自定义水汽臭氧成功 - TOA反射率: {s2.outputs.apparent_reflectance:.6f}")
        print(f"水汽: {water} g/cm², 臭氧: {ozone} cm-atm")
    except Exception as e:
        print(f"自定义水汽臭氧失败: {e}")

    # 测试3: 不同水汽含量的影响
    print("\n3. 测试不同水汽含量的影响")
    water_values = [1.0, 2.0, 3.0]

    for water in water_values:
        s3 = SixS()
        s3.atmos_profile = AtmosProfile.UserWaterAndOzone(water, 0.3)
        s3.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
        s3.aot550 = 0.2
        s3.wavelength = Wavelength(0.64)

        s3.geometry = Geometry.User()
        s3.geometry.solar_z = 30
        s3.geometry.view_z = 0

        s3.altitudes = Altitudes()
        s3.altitudes.set_target_custom_altitude(0)
        s3.altitudes.set_sensor_satellite_level()

        try:
            s3.run()
            print(f"水汽={water}g/cm² - TOA反射率: {s3.outputs.apparent_reflectance:.6f}")
        except Exception as e:
            print(f"水汽={water}g/cm² 失败: {e}")

    # 测试4: 不同臭氧含量的影响
    print("\n4. 测试不同臭氧含量的影响")
    ozone_values = [0.2, 0.3, 0.4]

    for ozone in ozone_values:
        s4 = SixS()
        s4.atmos_profile = AtmosProfile.UserWaterAndOzone(2.0, ozone)
        s4.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
        s4.aot550 = 0.2
        s4.wavelength = Wavelength(0.64)

        s4.geometry = Geometry.User()
        s4.geometry.solar_z = 30
        s4.geometry.view_z = 0

        s4.altitudes = Altitudes()
        s4.altitudes.set_target_custom_altitude(0)
        s4.altitudes.set_sensor_satellite_level()

        try:
            s4.run()
            print(f"臭氧={ozone}cm-atm - TOA反射率: {s4.outputs.apparent_reflectance:.6f}")
        except Exception as e:
            print(f"臭氧={ozone}cm-atm 失败: {e}")

    print("\n" + "=" * 60)
    print("水汽和臭氧测试完成")
    print("=" * 60)


def test_closed_loop_with_water_ozone():
    """测试包含水汽和臭氧的闭合循环"""
    print("\n" + "=" * 60)
    print("测试包含水汽和臭氧的闭合循环")
    print("=" * 60)

    try:
        # 正向模拟
        s_forward = SixS()
        s_forward.wavelength = Wavelength(0.64)
        s_forward.atmos_profile = AtmosProfile.UserWaterAndOzone(2.0, 0.3)
        s_forward.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
        s_forward.aot550 = 0.2

        s_forward.geometry = Geometry.User()
        s_forward.geometry.solar_z = 30
        s_forward.geometry.view_z = 0

        s_forward.altitudes = Altitudes()
        s_forward.altitudes.set_target_custom_altitude(0)
        s_forward.altitudes.set_sensor_satellite_level()

        s_forward.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.2)
        s_forward.atmos_corr = AtmosCorr.NoAtmosCorr()

        s_forward.run()
        rho_toa = s_forward.outputs.apparent_reflectance
        print(f"正向模拟 - TOA反射率: {rho_toa:.6f}")

        # 反演模拟
        s_inversion = SixS()
        s_inversion.wavelength = Wavelength(0.64)
        s_inversion.atmos_profile = AtmosProfile.UserWaterAndOzone(2.0, 0.3)
        s_inversion.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
        s_inversion.aot550 = 0.2

        s_inversion.geometry = Geometry.User()
        s_inversion.geometry.solar_z = 30
        s_inversion.geometry.view_z = 0

        s_inversion.altitudes = Altitudes()
        s_inversion.altitudes.set_target_custom_altitude(0)
        s_inversion.altitudes.set_sensor_satellite_level()

        s_inversion.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(rho_toa)

        s_inversion.run()

        print(f"反演结果 - 地表反射率: {s_inversion.outputs.pixel_reflectance:.6f}")
        print(f"误差: {s_inversion.outputs.pixel_reflectance - 0.2:.6f}")

        return True

    except Exception as e:
        print(f"闭合循环测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_water_ozone()
    test_closed_loop_with_water_ozone()