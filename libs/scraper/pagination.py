"""
Pagination Support for Job Scrapers

This module provides utilities for handling multi-page job listings in scrapers,
including pagination detection, navigation, and state management.

NOTE: This module requires selenium:
    pip install selenium
"""
from __future__ import annotations
import time
import logging
from typing import Optional, List, Dict, Any, Callable, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Optional selenium imports - gracefully handle missing dependencies  
try:
    from selenium.webdriver.remote.webdriver import WebDriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(f"Selenium not available: {e}. Install with 'pip install selenium'")
    SELENIUM_AVAILABLE = False
    # Create stub classes
    class WebDriver: pass
    class TimeoutException(Exception): pass
    class NoSuchElementException(Exception): pass

logger = logging.getLogger(__name__)


@dataclass
class PaginationState:
    """Tracks pagination state during scraping"""
    current_page: int = 1
    total_pages: Optional[int] = None
    has_next: bool = True
    page_urls: List[str] = field(default_factory=list)
    jobs_per_page: List[int] = field(default_factory=list)
    last_error: Optional[str] = None


@dataclass
class PaginationConfig:
    """Configuration for pagination behavior"""
    max_pages: int = 10  # Maximum pages to scrape
    wait_timeout: int = 10  # Timeout for page load waits
    page_delay: float = 2.0  # Delay between page navigations
    retry_failed_pages: bool = True  # Retry failed page loads
    detect_infinite_scroll: bool = False  # Handle infinite scroll pagination


class PaginationHandler(ABC):
    """Abstract base class for pagination handlers"""
    
    @abstractmethod
    def detect_pagination(self, driver: WebDriver) -> bool:
        """Check if pagination is present on the page"""
        pass
    
    @abstractmethod
    def get_total_pages(self, driver: WebDriver) -> Optional[int]:
        """Get total number of pages if available"""
        pass
    
    @abstractmethod
    def has_next_page(self, driver: WebDriver) -> bool:
        """Check if there's a next page available"""
        pass
    
    @abstractmethod
    def navigate_to_next_page(self, driver: WebDriver) -> bool:
        """Navigate to the next page, return True if successful"""
        pass
    
    @abstractmethod
    def navigate_to_page(self, driver: WebDriver, page_num: int) -> bool:
        """Navigate to a specific page number"""
        pass


class StandardPaginationHandler(PaginationHandler):
    """Handler for standard numbered pagination (1, 2, 3, ...)"""
    
    def __init__(self):
        # Common selectors for pagination elements
        self.pagination_selectors = [
            ".pagination",
            ".pager",
            ".page-navigation",
            "[class*='pagination']",
            "[class*='pager']",
            ".jobs-search-results-list__pagination"  # LinkedIn-style
        ]
        
        self.next_selectors = [
            "a[aria-label='Next']",
            "a[aria-label='Next page']",
            ".pagination-next",
            ".next-page",
            "a:contains('Next')",
            "a:contains('→')",
            "button:contains('Next')"
        ]
        
        self.page_info_selectors = [
            ".pagination-info",
            ".page-count",
            ".results-count",
            "[class*='page-info']"
        ]
    
    def detect_pagination(self, driver: WebDriver) -> bool:
        """Check if pagination elements are present"""
        try:
            for selector in self.pagination_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements and elements[0].is_displayed():
                    logger.debug(f"Found pagination with selector: {selector}")
                    return True
            return False
        except Exception as e:
            logger.debug(f"Error detecting pagination: {e}")
            return False
    
    def get_total_pages(self, driver: WebDriver) -> Optional[int]:
        """Extract total page count from pagination elements"""
        try:
            # Look for page numbers in pagination
            page_elements = driver.find_elements(By.CSS_SELECTOR, 
                ".pagination a, .pagination button, .pagination span")
            
            if page_elements:
                page_numbers = []
                for element in page_elements:
                    text = element.text.strip()
                    if text.isdigit():
                        page_numbers.append(int(text))
                
                if page_numbers:
                    return max(page_numbers)
            
            # Look for "Page X of Y" text
            for selector in self.page_info_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    text = element.text.lower()
                    if "of" in text and "page" in text:
                        # Extract "Page X of Y" pattern
                        words = text.split()
                        for i, word in enumerate(words):
                            if word == "of" and i + 1 < len(words):
                                try:
                                    return int(words[i + 1])
                                except ValueError:
                                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Error getting total pages: {e}")
            return None
    
    def has_next_page(self, driver: WebDriver) -> bool:
        """Check if next page button/link exists and is enabled"""
        try:
            for selector in self.next_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if (element.is_displayed() and 
                        element.is_enabled() and
                        "disabled" not in element.get_attribute("class")):
                        return True
            
            # Also check for numbered pagination - if we can find a higher number
            current_page = self._get_current_page(driver)
            if current_page:
                next_page_selector = f"a:contains('{current_page + 1}')"
                elements = driver.find_elements(By.CSS_SELECTOR, next_page_selector)
                return len(elements) > 0
                
            return False
            
        except Exception as e:
            logger.debug(f"Error checking for next page: {e}")
            return False
    
    def navigate_to_next_page(self, driver: WebDriver) -> bool:
        """Click the next page button/link"""
        try:
            for selector in self.next_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if (element.is_displayed() and 
                        element.is_enabled() and
                        "disabled" not in element.get_attribute("class")):
                        
                        current_url = driver.current_url
                        element.click()
                        
                        # Wait for page to load
                        WebDriverWait(driver, 10).until(
                            lambda d: d.current_url != current_url or 
                            self._page_content_changed(d, current_url)
                        )
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error navigating to next page: {e}")
            return False
    
    def navigate_to_page(self, driver: WebDriver, page_num: int) -> bool:
        """Navigate to a specific page number"""
        try:
            # Look for page number link
            page_link_selectors = [
                f"a:contains('{page_num}')",
                f"button:contains('{page_num}')",
                f".pagination a[data-page='{page_num}']",
                f".pagination button[data-page='{page_num}']"
            ]
            
            for selector in page_link_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if element.is_displayed() and element.is_enabled():
                        current_url = driver.current_url
                        element.click()
                        
                        # Wait for navigation
                        WebDriverWait(driver, 10).until(
                            lambda d: d.current_url != current_url or
                            self._page_content_changed(d, current_url)
                        )
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error navigating to page {page_num}: {e}")
            return False
    
    def _get_current_page(self, driver: WebDriver) -> Optional[int]:
        """Get the current page number"""
        try:
            # Look for active/current page indicators
            active_selectors = [
                ".pagination .active",
                ".pagination .current",
                ".pagination [aria-current='page']"
            ]
            
            for selector in active_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    text = element.text.strip()
                    if text.isdigit():
                        return int(text)
            
            return None
            
        except Exception:
            return None
    
    def _page_content_changed(self, driver: WebDriver, original_url: str) -> bool:
        """Check if page content has changed (for AJAX pagination)"""
        # Simple check - in practice might need more sophisticated detection
        return driver.current_url != original_url


class InfiniteScrollHandler(PaginationHandler):
    """Handler for infinite scroll pagination"""
    
    def __init__(self, max_scrolls: int = 20):
        self.max_scrolls = max_scrolls
        self.scroll_pause_time = 2
    
    def detect_pagination(self, driver: WebDriver) -> bool:
        """Check if infinite scroll is likely present"""
        # Look for indicators of infinite scroll
        scroll_indicators = [
            "[data-infinite-scroll]",
            ".infinite-scroll",
            ".lazy-load",
            "[class*='infinite']",
            "[class*='lazy-load']"
        ]
        
        for selector in scroll_indicators:
            if driver.find_elements(By.CSS_SELECTOR, selector):
                return True
        
        # Check if scrolling reveals more content
        initial_height = driver.execute_script("return document.body.scrollHeight")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        return new_height > initial_height
    
    def get_total_pages(self, driver: WebDriver) -> Optional[int]:
        """Infinite scroll doesn't have discrete pages"""
        return None
    
    def has_next_page(self, driver: WebDriver) -> bool:
        """Check if more content can be loaded by scrolling"""
        current_height = driver.execute_script("return document.body.scrollHeight")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(self.scroll_pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        return new_height > current_height
    
    def navigate_to_next_page(self, driver: WebDriver) -> bool:
        """Scroll to load more content"""
        return self.has_next_page(driver)
    
    def navigate_to_page(self, driver: WebDriver, page_num: int) -> bool:
        """Not applicable for infinite scroll"""
        return False


def paginate_scraper(
    scraper_function: Callable[[WebDriver], List[Any]],
    driver: WebDriver,
    config: PaginationConfig,
    pagination_handler: Optional[PaginationHandler] = None
) -> Iterator[List[Any]]:
    """
    Generator that handles pagination for a scraper function
    
    Args:
        scraper_function: Function that scrapes jobs from current page
        driver: WebDriver instance
        config: Pagination configuration
        pagination_handler: Custom pagination handler (uses StandardPaginationHandler if None)
        
    Yields:
        Lists of scraped items from each page
    """
    if pagination_handler is None:
        pagination_handler = StandardPaginationHandler()
    
    state = PaginationState()
    
    # Check if pagination exists
    if not pagination_handler.detect_pagination(driver):
        logger.info("No pagination detected, scraping single page")
        yield scraper_function(driver)
        return
    
    # Get total pages if available
    state.total_pages = pagination_handler.get_total_pages(driver)
    if state.total_pages:
        logger.info(f"Detected {state.total_pages} total pages")
        config.max_pages = min(config.max_pages, state.total_pages)
    
    # Scrape pages
    for page_num in range(1, config.max_pages + 1):
        try:
            logger.info(f"Scraping page {page_num}")
            state.current_page = page_num
            
            # Scrape current page
            page_jobs = scraper_function(driver)
            state.jobs_per_page.append(len(page_jobs))
            state.page_urls.append(driver.current_url)
            
            yield page_jobs
            
            # Check if we should continue
            if not pagination_handler.has_next_page(driver):
                logger.info("No more pages available")
                break
            
            # Navigate to next page
            if page_num < config.max_pages:
                success = pagination_handler.navigate_to_next_page(driver)
                if not success:
                    logger.warning(f"Failed to navigate to page {page_num + 1}")
                    if not config.retry_failed_pages:
                        break
                
                # Wait between pages
                time.sleep(config.page_delay)
            
        except Exception as e:
            error_msg = f"Error scraping page {page_num}: {e}"
            logger.error(error_msg)
            state.last_error = error_msg
            
            if not config.retry_failed_pages:
                break
    
    logger.info(f"Pagination complete. Scraped {state.current_page} pages, "
                f"total jobs: {sum(state.jobs_per_page)}")