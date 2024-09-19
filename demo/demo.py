import os
import json
from multiprocessing import Pool
from loguru import logger

from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.model.crop_image_multiprocess import load_images_from_pdf
import magic_pdf.model as model_config
model_config.__use_inside_model__ = True


if __name__ == '__main__':
    import time
    try:
        images = None
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        demo_name = "demo1"
        pdf_path = os.path.join(current_script_dir, f"{demo_name}.pdf")
        model_path = os.path.join(current_script_dir, f"{demo_name}.json")
        pdf_bytes = open(pdf_path, "rb").read()
        start_time = time.time()
        pool = Pool(processes=2)
        images = load_images_from_pdf(pdf_bytes, dpi=200, pool=pool)
        end_time = time.time()
        logger.info(f"images crop speed: {len(pdf_bytes)/(end_time-start_time)} pages/s")
        logger.info(f"images crop cost: {(end_time - start_time)} s")
        # print(f"===========================================image length: {len(images)}")
        # model_json = json.loads(open(model_path, "r", encoding="utf-8").read())
        model_json = []  # model_json传空list使用内置模型解析
        jso_useful_key = {"_pdf_type": "", "model_list": model_json}
        local_image_dir = os.path.join(current_script_dir, 'images')
        image_dir = str(os.path.basename(local_image_dir))
        image_writer = DiskReaderWriter(local_image_dir)

        pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer, images=images)
        pipe.pipe_classify()
        """如果没有传入有效的模型数据，则使用内置model解析"""
        if len(model_json) == 0:
            if model_config.__use_inside_model__:
                pipe.pipe_analyze()
            else:
                logger.error("need model list input")
                exit(1)
        pipe.pipe_parse()
        md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
        with open(f"{demo_name}.md", "w", encoding="utf-8") as f:
            f.write(md_content)
    except Exception as e:
        logger.exception(e)
    finally:
        pool.close()
        pool.join()