import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from multiprocessing import Pool
import logging
logger = logging.getLogger(__name__)


def process_page(args):
    """
    处理单个页面的函数，用于多进程调用
    参数为一个元组 (page_index, pdf_bytes, dpi)，其中page_index是页面索引，pdf_bytes是PDF的字节流，dpi为分辨率
    """
    page_index, pdf_bytes, dpi = args

    # 在子进程中重新打开PDF文档
    with fitz.open("pdf", pdf_bytes) as doc:
        page = doc[page_index]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pm = page.get_pixmap(matrix=mat, alpha=False)

        # 如果宽度或高度在缩放后超过9000，则不再进一步缩放
        if pm.width > 9000 or pm.height > 9000:
            pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

        img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
        img = np.array(img)
        img_dict = {"image_id": page_index, "img": img, "width": pm.width, "height": pm.height}
    return img_dict


def load_images_from_pdf(pdf_bytes: bytes, dpi=200, pool=None) -> list:
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow not installed, please install by pip.")
        exit(1)

    # 使用上下文管理器打开PDF文档一次，获取总页数
    with fitz.open("pdf", pdf_bytes) as doc:
        num_pages = doc.page_count

        # 创建一个固定大小的进程池
        # with Pool(processes=min(max_worker, num_pages)) as pool:
            # 准备参数列表，每个元素都是一个元组 (page_index, pdf_bytes, dpi)
        page_args = [(i, pdf_bytes, dpi) for i in range(num_pages)]

        # 使用map应用process_page函数到每个页面上
        images = pool.map(process_page, page_args)
    images = sorted(images, key=lambda x:x["image_id"])
    return images

if __name__ == '__main__':
    input = "D:\\projects\\Magic-PDF\\demo\\demo1.pdf"
    with open(input, 'rb') as f:
        pdf_bytes = f.read()


        # 调用函数加载图片
    images = load_images_from_pdf(pdf_bytes, dpi=200)

    # 输出结果（这里简单输出到控制台）
    for img in images:
        print(f"Image size: {img['width']}x{img['height']}, DPI: 200")