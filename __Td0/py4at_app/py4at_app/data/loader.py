"""
Data Loader Module
Handles retrieval and preparation of financial data
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class DataLoader:
    """
    Utility class for loading and preparing financial data.
    
    Methods
    =======
    load_from_csv:
        Load financial data from CSV file
    load_from_url:
        Load financial data from URL
    prepare_data:
        Prepare and calculate returns and technical indicators
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def load_from_csv(filepath: str, symbol: Optional[str] = None,
                      start: Optional[str] = None, 
                      end: Optional[str] = None) -> pd.DataFrame:
        """
        Load financial data from a CSV file.
        
        Parameters
        ==========
        filepath: str
            Path to the CSV file
        symbol: str, optional
            Symbol/column to extract (if None, uses first data column)
        start: str, optional
            Start date for filtering (format: 'YYYY-MM-DD')
        end: str, optional
            End date for filtering (format: 'YYYY-MM-DD')
            
        Returns
        =======
        pd.DataFrame
            DataFrame with price data
        """
        raw = pd.read_csv(filepath, index_col=0, parse_dates=True).dropna()
        
        if symbol:
            raw = pd.DataFrame(raw[symbol])
        else:
            raw = pd.DataFrame(raw.iloc[:, 0])
        
        raw.columns = ['price']
        
        if start and end:
            raw = raw.loc[start:end]
        elif start:
            raw = raw.loc[start:]
        elif end:
            raw = raw.loc[:end]
            
        return raw
    
    @staticmethod
    def load_from_url(url: str, symbol: str,
                      start: Optional[str] = None,
                      end: Optional[str] = None) -> pd.DataFrame:
        """
        Load financial data from a URL.
        
        Parameters
        ==========
        url: str
            URL to fetch data from
        symbol: str
            Symbol/column to extract
        start: str, optional
            Start date for filtering
        end: str, optional
            End date for filtering
            
        Returns
        =======
        pd.DataFrame
            DataFrame with price data
        """
        raw = pd.read_csv(url, index_col=0, parse_dates=True).dropna()
        raw = pd.DataFrame(raw[symbol])
        raw.columns = ['price']
        
        if start and end:
            raw = raw.loc[start:end]
        elif start:
            raw = raw.loc[start:]
        elif end:
            raw = raw.loc[:end]
            
        return raw
    
    @staticmethod
    def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data by calculating returns.
        
        Parameters
        ==========
        data: pd.DataFrame
            DataFrame with 'price' column
            
        Returns
        =======
        pd.DataFrame
            DataFrame with added 'return' column (log returns)
        """
        data = data.copy()
        data['return'] = np.log(data['price'] / data['price'].shift(1))
        return data.dropna()
    
    @staticmethod
    def add_sma(data: pd.DataFrame, window: int, 
                column: str = 'price') -> pd.DataFrame:
        """
        Add Simple Moving Average to data.
        
        Parameters
        ==========
        data: pd.DataFrame
            Input DataFrame
        window: int
            SMA window size
        column: str
            Column to calculate SMA on
            
        Returns
        =======
        pd.DataFrame
            DataFrame with added SMA column
        """
        data = data.copy()
        data[f'SMA_{window}'] = data[column].rolling(window).mean()
        return data
    
    @staticmethod
    def add_momentum(data: pd.DataFrame, window: int = 1) -> pd.DataFrame:
        """
        Add momentum indicator to data.
        
        Parameters
        ==========
        data: pd.DataFrame
            Input DataFrame with 'return' column
        window: int
            Window size for momentum calculation
            
        Returns
        =======
        pd.DataFrame
            DataFrame with added 'momentum' column
        """
        data = data.copy()
        data['momentum'] = np.sign(data['return'].rolling(window).mean())
        return data
