import numpy as np
import pandas as pd

class PostProcessor:
    def __init__(self):
        self.cruise_min_alt = 15.0
        self.heading_buffer = 30.0

    def get_flight_states(self, df, vline_times, labels, time_col='SHIFTED_TIME'):
        """Classifies global flight phases (STATE) using time bins."""
        df = df.copy()
        df['STATE'] = pd.cut(
            df[time_col],
            bins=[-np.inf] + vline_times + [np.inf],
            labels=labels,
            right=False
        )
        df['SUBSTATE'] = 'OUTBOUND'
        return df

    def classify_substate_auto(self, df, cruise_segment, heading_col='HEADING', 
                              state_col='STATE', time_col='SHIFTED_TIME', threshold=30):
        """
        Automated heading-based SUBSTATE classification.
        Uses the first heading of the segment as the Outbound reference.
        """
        mask = df[state_col] == cruise_segment
        if not mask.any(): return df
        
        segment_data = df[mask].sort_values(time_col)
        ref_h = segment_data[heading_col].iloc[0]
        in_h = (ref_h + 180) % 360
        
        headings = df.loc[mask, heading_col]
        
        # Calculate circular differences
        out_diff = np.minimum(abs(headings - ref_h), 360 - abs(headings - ref_h))
        in_diff = np.minimum(abs(headings - in_h), 360 - abs(headings - in_h))
        
        df.loc[mask, 'SUBSTATE'] = np.where(out_diff <= threshold, 'OUTBOUND',
                                   np.where(in_diff <= threshold, 'INBOUND', 'TURNING'))
        return df

    def classify_substate_freq(self, df, cruise_segment, heading_col='HEADING', 
                               state_col='STATE', time_col='SHIFTED_TIME', threshold=35):
        """
        Automated classification using the most frequent heading of the first 30% 
        (useful for segments starting with turns).
        """
        mask = df[state_col] == cruise_segment
        if not mask.any(): return df
        
        segment_data = df[mask].sort_values(time_col)
        early_data = segment_data.head(int(len(segment_data) * 0.3))
        
        # Histogram to find dominant heading
        hist, edges = np.histogram(early_data[heading_col], bins=np.arange(0, 361, 10))
        ref_h = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
        in_h = (ref_h + 180) % 360
        
        headings = df.loc[mask, heading_col]
        out_diff = np.minimum(abs(headings - ref_h), 360 - abs(headings - ref_h))
        in_diff = np.minimum(abs(headings - in_h), 360 - abs(headings - in_h))
        
        df.loc[mask, 'SUBSTATE'] = np.where(out_diff <= threshold, 'OUTBOUND',
                                   np.where(in_diff <= threshold, 'INBOUND', 'TURNING'))
        return df

    def update_substate_manual(self, df, segment, start, end, new_substate, 
                               state_col='STATE', time_col='SHIFTED_TIME'):
        """Manual time-based override for complex turns."""
        mask = (df[state_col] == segment) & (df[time_col] >= start) & (df[time_col] <= end)
        df.loc[mask, 'SUBSTATE'] = new_substate
        return df

processor = PostProcessor()