from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import time

# Path to your ChromeDriver executable
chromedriver_path = "C://Users//JEFFREY//Downloads//chromedriver-win64//chromedriver-win64//chromedriver.exe"

# Create a ChromeDriver service
service = Service(chromedriver_path)

# Set Chrome options to run in headless mode
options = webdriver.ChromeOptions()
options.add_argument("--no-sandbox")
# options.add_argument("--headless")  # Enable headless mode
# options.add_argument("--disable-gpu")  # Disable GPU for headless mode
# options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems

# Initialize the driver
driver = webdriver.Chrome(service=service, options=options)

try:
    # Open Google Maps
    url = "https://www.google.com/maps"
    driver.get(url)

    while True:
        # Wait for the page to load
        time.sleep(5)  # Adjust the sleep duration as needed

        # Attempt to click the "Show Your Location" button using explicit wait
        try:
            button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, 'sVuEFc'))
            )
            button.click()
            print("Clicked the 'Show Your Location' button.")
        except Exception as e:
            print("Could not click the 'Show Your Location' button:", e)

        # Wait for a bit to allow the map to center on the current location
        time.sleep(5)

        # Get the current URL
        current_url = driver.current_url
        print(f'Current URL: {current_url}')

        # Use a regular expression to extract the latitude and longitude
        match = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', current_url)
        if match:
            latitude = match.group(1)
            longitude = match.group(2)
            print(f'Latitude: {latitude}, Longitude: {longitude}')
        else:
            print("Coordinates not found in the URL")

        # Wait before refreshing the page
        time.sleep(10)  # Adjust the sleep duration as needed
        driver.refresh()

finally:
    # Close the browser
    driver.quit()
