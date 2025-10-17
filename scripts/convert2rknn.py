from rknn.api import RKNN
import os

if __name__ == "__main__":
    platform = "rk3588s"

    """step 1: create RKNN object"""
    rknn = RKNN()

    """step 2: load the .onnx model"""
    rknn.config(target_platform="rk3588s")
    # day='2025-09-26_15-32-36_ticao_80_frict'
    day='2025-10-07_16-28-57_kick1_80_COM'

   
    # pose='hi_flat'
    pose='pi_plus_flat'

  


    file_dir=f'/home/youyou/bymimic_hi/logs/rsl_rl/{pose}/{day}/exported'
    model_version= 'model_30000_action'
    file_name=f'{model_version}.onnx'
    file_path=os.path.join(file_dir,file_name)

    print("--> Loading model")
    ret = rknn.load_onnx(file_path)
    if ret != 0:
        print("load model failed")
        exit(ret)
    print("done")

    """step 3: building model"""
    print("-->Building model")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("build model failed")
        exit()
    print("done")

    """step 4: export and save the .rknn model"""
    OUT_DIR = f"/home/youyou//bymimic_hi/logs/rsl_rl/{pose}/{day}/exported"
    RKNN_MODEL_PATH = f"/{OUT_DIR}/{model_version}_{day}.rknn"
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    print("--> Export RKNN model: {}".format(RKNN_MODEL_PATH))
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print("Export rknn model failed.")
        exit(ret)
    print("done")

    """step 5: release the model"""
    rknn.release()
