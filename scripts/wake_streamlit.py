import os

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

WAKE_BUTTON_XPATH = "//button[contains(., 'Yes, get this app back up')]"
APP_READY_XPATH = "//div[@data-testid='stAppViewContainer']"


def wake_app() -> None:
    url = os.environ["STREAMLIT_URL"]

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )

    try:
        print(f"访问: {url}")
        driver.get(url)

        try:
            wake_button = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.XPATH, WAKE_BUTTON_XPATH))
            )
            wake_button.click()
            print("点击了唤醒按钮")
            WebDriverWait(driver, 120).until(
                EC.presence_of_element_located((By.XPATH, APP_READY_XPATH))
            )
            print("应用已唤醒并加载")
        except TimeoutException:
            print("未发现唤醒按钮，应用可能本来就是活跃的")
    finally:
        driver.quit()


if __name__ == "__main__":
    wake_app()
