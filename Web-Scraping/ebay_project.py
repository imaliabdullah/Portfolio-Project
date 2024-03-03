from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import pandas as pd


website = "https://www.ebay.com/sch/i.html?_from=R40&_trksid=p4432023.m570.l1313&_nkw=drones&_sacat=0"

driver = webdriver.Chrome()
driver.get(website)

item_names = []
prices = []
links = []

# Set the maximum number of pages to scrape
max_pages = 3
current_page = 1

while current_page <= max_pages:
    try:
        # Locate and process products on the current page
        products = driver.find_elements(by='xpath', value='//li[contains(@class, "s-item")]')
        for product in products:
            item_name = product.find_element(by='xpath', value=".//div[contains(@class, 's-item__title')]").text
            price = product.find_element(by='xpath', value=".//span[contains(@class, 's-item__price')]").text
            link = product.find_element(by='xpath', value=".//div[contains(@class, 's-item__info')]/a").get_attribute('href')

            item_names.append(item_name)
            prices.append(price)
            links.append(link)

            print(item_name)
            print(price)
            print(link)

        # Increment the page counter
        current_page += 1

        # Locate the next page button and click on it
        next_page_button = driver.find_element(by='xpath', value='//ol[@class="pagination__items"]/li/a[@href]')
        next_page_button.click()

    except NoSuchElementException:
        # If no next page button is found, break out of the loop
        break

driver.quit()

df2 = pd.DataFrame({'Item Name': item_names, 'Price': prices, 'Links': links})

df2 = df2.drop(df2.index[0])
df2.to_csv('ebaylinks.csv')