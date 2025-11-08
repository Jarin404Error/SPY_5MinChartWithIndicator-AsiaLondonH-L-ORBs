import yfinance as yf
import pandas as pd
import datetime as dt
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
# import os  # Okay, commenting out the AI stuff for now.
# from google import genai  # This was for the Gemini API call.
# from dotenv import load_dotenv  # Need this for the API key, will add it back later.

# --- Configuration ---
TICKER = "SPY"
MARKET_TZ = "America/New_York"  # NYSE Timezone

# --- Session Times in ET ---
ASIA_SESSION_START = dt.time(20, 0) # 8:00 PM
ASIA_SESSION_END = dt.time(4, 0)    # 4:00 AM

LONDON_SESSION_START = dt.time(3, 0) # 3:00 AM
LONDON_SESSION_END = dt.time(11, 30) # 11:30 AM

# --- US Market Times in ET ---
PREMARKET_OPEN = dt.time(4, 0)
MARKET_OPEN = dt.time(9, 30)
ORB_5_END = dt.time(9, 35)
ORB_15_END = dt.time(9, 45) # 9:30 + 15 mins
MARKET_CLOSE = dt.time(16, 0)

def get_level_colors():
    """Returns a consistent color mapping for levels."""
    # Just a helper function to keep the chart colors consistent
    return {
        "PM_High": "red",
        "PM_Low": "red",
        "ORB_5_High": "blue",
        "ORB_5_Low": "blue",
        "ORB_15_High": "green",
        "ORB_15_Low": "green",
        "Asia_Session_High": "purple",
        "Asia_Session_Low": "purple",
        "London_Session_High": "orange",
        "London_Session_Low": "orange",
        "PDH": "gray", # Previous Day High
        "PDL": "gray", # Previous Day Low
    }

# --- This is where I was trying to add the Gemini AI analysis ---
# --- (Commenting it out for now, it's a future project) ---
# I want to feed the levels to the AI and get a summary.
# Need to get the premium API keys working first.

# def get_gemini_analysis(levels, current_price, ticker):
#     """
#     Sends the current market data to the Gemini API for a dynamic analysis.
#     """
#     print("Getting dynamic Gemini AI analysis...")
#     try:
#         load_dotenv() # This should load GOOGLE_API_KEY from my .env file
#         
#         # 1. Configure the API key
#         api_key = os.getenv("GOOGLE_API_KEY")
#         if not api_key:
#             # This debug message was super helpful
#             print("--- DEBUG: FAILED TO LOAD GOOGLE_API_KEY! ---")
#             print("--- Make sure .env file has GOOGLE_API_KEY=... ---")
#             return "<p><strong>Could not retrieve Gemini analysis.</strong> Is your GOOGLE_API_KEY set in the .env file?</p>"
#         
#         print("--- DEBUG: Gemini API Key loaded successfully. ---")
#         genai.configure(api_key=api_key)
#         
#         # 2. Format the data into a clean string for the AI
#         # Just making a simple string for the prompt
#         level_summary = "\n".join([f"  {name}: {price:.2f}" for name, price in levels.items() if price])
#         
#         # 3. Set up the model configuration
#         # Copied these from the docs, seemed reasonable
#         generation_config = {
#           "temperature": 0.5, # Not too creative
#           "top_p": 1,
#           "top_k": 1,
#           "max_output_tokens": 250, # Keep it brief
#         }
# 
#         # Standard safety settings
#         safety_settings = [
#           {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#           {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#           {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#           {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#         ]
#         
#         model = genai.GenerativeModel(
#             model_name="gemini-pro", # This model is fast and cheap
#             generation_config=generation_config,
#             safety_settings=safety_settings
#         )
#         
#         # 4. Create the user prompt
#         # The prompt engineering part is tricky. Trying to make it neutral.
#         # Explicitly telling it NOT to give advice.
#         prompt = f"""
#         You are a neutral, educational trading assistant. 
#         You NEVER give financial advice, predictions, or recommendations (e.g., "buy calls", "sell puts", "the price will go up").
#         Your role is to analyze the provided market data and give a brief, educational summary
#         of the current situation. Focus on the price's relationship to the key levels.
#         Keep your response to 3-4 short bullet points or a short paragraph.
# 
#         Here is the current market data for {ticker}:
# 
#         Current Price: {current_price:.2f}
# 
#         Key Levels:
#         {level_summary}
# 
#         Please provide a brief, educational analysis of the current situation. 
#         What might a trader be observing? 
#         (REMEMBER: Do not give any advice or predictions).
#         """
#         
#         # 5. Make the API Call
#         print("--- DEBUG: Sending prompt to Gemini... ---")
#         response = model.generate_content(prompt)
#         
#         # Return the AI's text response
#         return response.text
#         
#     except Exception as e:
#         print(f"Error getting Gemini analysis: {e}")
#         return "<p><strong>Could not retrieve Gemini analysis.</strong> An error occurred.</p>"
# --- (End of the AI analysis function) ---


def calculate_levels(df_2day):
    """Calculates all key levels from the 2-day dataframe."""
    levels = {}
    
    # --- Robust Date Calculation ---
    # Had a bug here before, np.unique fixed it.
    all_dates = sorted(np.unique(df_2day.index.date)) 
    
    if len(all_dates) == 0:
        print("No dates found in data.")
        return {}
    
    today_date = all_dates[-1] # The most recent date is "today"
    prev_trading_date = None
    if len(all_dates) > 1:
        prev_trading_date = all_dates[-2] # The second to last date
    # --- End Robust Date Calculation ---

    # 0. Previous Day High/Low
    if prev_trading_date:
        try:
            df_prev_day = df_2day[df_2day.index.date == prev_trading_date]
            # Need to make sure I'm only getting RTH (Regular Trading Hours)
            df_prev_day_rth = df_prev_day.between_time(MARKET_OPEN, MARKET_CLOSE)
            
            if not df_prev_day_rth.empty:
                levels["PDH"] = df_prev_day_rth['High'].max()
                levels["PDL"] = df_prev_day_rth['Low'].min()
            else:
                # Fallback just in case, like on a weird holiday
                print("Warning: No RTH data for prev day. Using full day.")
                if not df_prev_day.empty:
                    levels["PDH"] = df_prev_day['High'].max()
                    levels["PDL"] = df_prev_day['Low'].min()
        except Exception as e:
            print(f"Could not calculate Previous Day levels: {e}")

    # 1. Asia Session Levels (Last night to this morning)
    if prev_trading_date: # Only run if we have a previous day
        try:
            # This logic is tricky, spans from yesterday evening to this morning
            asia_start_dt = pytz.timezone(MARKET_TZ).localize(dt.datetime.combine(prev_trading_date, ASIA_SESSION_START))
            asia_end_dt = pytz.timezone(MARKET_TZ).localize(dt.datetime.combine(today_date, ASIA_SESSION_END))
            
            df_asia = df_2day.loc[asia_start_dt : asia_end_dt]
            if not df_asia.empty:
                levels["Asia_Session_High"] = df_asia['High'].max()
                levels["Asia_Session_Low"] = df_asia['Low'].min()
        except Exception as e:
            print(f"Could not calculate Asia session levels: {e}")
    else:
        print("Not enough data to calculate previous day's (Asia) session.")

    # 2. London Session Levels (Early this morning)
    try:
        # This one is all on today_date
        london_start_dt = pytz.timezone(MARKET_TZ).localize(dt.datetime.combine(today_date, LONDON_SESSION_START))
        london_end_dt = pytz.timezone(MARKET_TZ).localize(dt.datetime.combine(today_date, LONDON_SESSION_END))
        
        df_london = df_2day.loc[london_start_dt : london_end_dt]
        if not df_london.empty:
            levels["London_Session_High"] = df_london['High'].max()
            levels["London_Session_Low"] = df_london['Low'].min()
    except Exception as e:
        print(f"Could not calculate London session levels: {e}")

    # 3. Pre-Market Levels (Today)
    df_premarket = df_2day.between_time(PREMARKET_OPEN, MARKET_OPEN)
    if not df_premarket.empty:
        # Need to filter *again* for today_date, or it'll grab yesterday's PM too
        df_premarket = df_premarket[df_premarket.index.date == today_date]
        if not df_premarket.empty:
            levels["PM_High"] = df_premarket['High'].max()
            levels["PM_Low"] = df_premarket['Low'].min()

    # 4. 5-min ORB Levels (Today)
    df_orb5 = df_2day.between_time(MARKET_OPEN, ORB_5_END)
    if not df_orb5.empty:
        df_orb5 = df_orb5[df_orb5.index.date == today_date]
        if not df_orb5.empty:
            levels["ORB_5_High"] = df_orb5['High'].max()
            levels["ORB_5_Low"] = df_orb5['Low'].min()

    # 5. 15-min ORB Levels (Today)
    df_orb15 = df_2day.between_time(MARKET_OPEN, ORB_15_END)
    if not df_orb15.empty:
        df_orb15 = df_orb15[df_orb15.index.date == today_date]
        if not df_orb15.empty:
            levels["ORB_15_High"] = df_orb15['High'].max()
            levels["ORB_15_Low"] = df_orb15['Low'].min()
            
    return levels

def plot_chart(df, levels, current_price):
    """Creates an interactive Plotly chart with levels."""
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        subplot_titles=(f'{TICKER} 5-Min Chart', 'Volume'), 
        row_heights=[0.8, 0.2] # More space for the price chart
    )
    
    chart_date = df.index[0].date()
    chart_title = f"{TICKER} Levels - {chart_date.strftime('%Y-%m-%d')}"

    # --- Add Shaded Pre-Market Region ---
    pm_start_dt = df.index[0] 
    pm_end_dt = pytz.timezone(MARKET_TZ).localize(dt.datetime.combine(chart_date, MARKET_OPEN))

    fig.add_vrect(
        x0=pm_start_dt, x1=pm_end_dt,
        fillcolor="rgba(100, 100, 100, 0.15)",
        layer="below", # Draw it behind the candles
        line_width=0,
        row=1, col=1
    )
    
    fig.add_annotation(
        x=pm_start_dt + (pm_end_dt - pm_start_dt) / 2, # Center the text
        y=df['High'].max(),
        yanchor="top",
        text="Pre-Market Session",
        showarrow=False,
        font=dict(color="rgba(100, 100, 100, 0.6)"),
        row=1, col=1
    )

    # 1. Candlestick Chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candles'
        ),
        row=1, col=1
    )

    # 2. Volume Chart
    color_pos = 'rgba(0, 150, 0, 0.5)'
    color_neg = 'rgba(200, 0, 0, 0.5)'
    color_neu = 'rgba(100, 100, 100, 0.5)'
    
    # This numpy 'where' is so much cleaner than a loop
    colors = np.where(df['Close'] > df['Open'], color_pos, 
                            np.where(df['Close'] < df['Open'], color_neg, color_neu))

    fig.add_trace(
        go.Bar(
            x=df.index, 
            y=df['Volume'], 
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )

    # 3. Add Level Lines
    colors = get_level_colors()
    for level_name, level_price in levels.items():
        if level_price is None:
            continue
        
        fig.add_shape(
            type="line",
            x0=df.index[0], y0=level_price,
            x1=df.index[-1], y1=level_price,
            line=dict(
                color=colors.get(level_name, "gray"),
                width=1.5,
                dash="dash",
            ),
            name=f"{level_name} ({level_price:.2f})",
            row=1, col=1
        )
        
        # --- Annotations anchored to the LEFT side ---
        # This looks way cleaner than putting them in the legend
        fig.add_annotation(
            x=df.index[0],
            y=level_price,
            text=f"{level_name} ({level_price:.2f})",
            showarrow=False,
            xanchor="left",
            xshift=5,
            yanchor="bottom",
            font=dict(color=colors.get(level_name, "gray"), size=10),
            bgcolor="rgba(255, 255, 255, 0.7)", # So it's readable over candles
            row=1, col=1
        )

    # --- Customize Layout ---
    fig.update_layout(
        title=chart_title,
        xaxis_title="Time",
        yaxis_title="Price",
        height=800,
        showlegend=False, # Legend is redundant with the annotations
        xaxis_rangeslider_visible=False, # Annoying slider
        xaxis=dict(range=[df.index.min(), df.index.max()]),
        yaxis=dict(autorange=True, fixedrange=False), # Let user zoom Y-axis
        yaxis2=dict(autorange=True, fixedrange=False)
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.1)')

    # --- HTML Generation Idea ---
    # My plan here was to export the chart and build a custom HTML page.
    # The idea is to have the Plotly chart on the left and a sidebar on the right
    # with the AI analysis and maybe other data.
    # Commenting this out for now to just get the basic chart working.
    
    # <-- START of the HTML/AI sidebar code -->
    #
    # # This gets the chart as a single <div>
    # chart_div = fig.to_html(full_html=False, include_plotlyjs='cdn') 
    # 
    # # Here I'd call the AI function (which is also commented out)
    # ai_text = get_gemini_analysis(levels, current_price, TICKER)
    # 
    # # Then just simple string formatting to build the HTML sidebar
    # analysis_html = f"""
    #     <h3>Dynamic AI Analysis (Gemini)</h3>
    #     <p><strong>Current {TICKER} Price: ${current_price:.2f}</strong></p>
    #     <div>{ai_text.replace(chr(10), '<br>').replace('*', '')}</div>
    #     <br>
    #     <p>
    #         <strong>Disclaimer: This is not financial advice.</strong> All trading involves
    #         significant risk. This information is for educational purposes only.
    #     </p>
    # """
    # 
    # # A placeholder for options data, which is a whole other project
    # options_html = """
    #     <h3>Live Options Data</h3>
    #     <p>
    #         <strong>AI Limitation:</strong> I cannot fetch live options data in this view.
    #     </p>
    # """
    # 
    # # And finally, jam it all together into one big HTML string
    # html_content = f"""
    # <html>
    #     <head>
    #         <title>{chart_title}</title>
    #         <style>
    #             /* Basic CSS to make it not look terrible */
    #             body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #f9f9f9; }}
    #             .container {{ display: flex; flex-direction: row; padding: 20px; }}
    #             .chart-container {{ flex: 3; }}
    #             .sidebar {{ flex: 1; padding-left: 20px; border-left: 1px solid #ddd; }}
    #             h1 {{ text-align: center; }}
    #             h3 {{ border-bottom: 2px solid #eee; padding-bottom: 5px; }}
    #             .sidebar div {{ line-height: 1.6; }}
    #         </style>
    #     </head>
    #     <body>
    #         <h1>{chart_title}</h1>
    #         <div class="container">
    #             <div class="chart-container">
    #                 {chart_div}
    #             </div>
    #             <div class="sidebar">
    #                 {analysis_html}
    #                 <hr>
    #                 {options_html}
    #             </div>
    #         </div>
    #     </body>
    # </html>
    # """
    # 
    # # This would save it so I could open the file
    # chart_filename = f"{TICKER}_daily_chart.html"
    # try:
    #     with open(chart_filename, 'w', encoding='utf-8') as f:
    #         f.write(html_content)
    #     print(f"Chart saved as {chart_filename}")
    # except Exception as e:
    #     print(f"Error writing HTML file: {e}")
    #
    # <-- END of the HTML/AI sidebar code -->

    # For now, just show the chart in the browser like normal.
    fig.show() 


def main():
    """Main function to fetch data and generate chart."""
    print(f"Fetching {TICKER} data for today...")
    
    ticker = yf.Ticker(TICKER)
    
    data = ticker.history(
        period="2d", # 2 days to get previous day and overnight
        interval="5m", 
        prepost=True, # This is the key! Need pre/post market data
        auto_adjust=False, # Want the raw prices
        back_adjust=False  #
    )
    
    if data.empty:
        print("No data found. Is the market open today?")
        return
        
    # Always convert to market timezone
    data.index = data.index.tz_convert(MARKET_TZ)
    
    print("Calculating levels...")
    levels = calculate_levels(data)
    
    print("--- Calculated Levels ---")
    # Sorting them makes it easier to read
    sorted_levels = sorted(
        [item for item in levels.items() if item[1] is not None], 
        key=lambda item: item[1], 
        reverse=True
    )
    for name, price in sorted_levels:
        print(f"  {name:<20}: {price:.2f}")
    print("-------------------------")

    # --- Filter data for CHARTS ---
    # Only want to chart *today's* data, starting from premarket
    today_date = data.index[-1].date()
    today_start_dt = pytz.timezone(MARKET_TZ).localize(dt.datetime.combine(today_date, PREMARKET_OPEN))
    
    data_filtered = data.loc[today_start_dt:]
    
    if data_filtered.empty:
        print("No data found for today's session yet.")
        return

    # Get the very last price to send to the (future) AI
    current_price = data_filtered['Close'].iloc[-1]

    # Changed this, since the AI part is commented out
    print("Generating chart...") 
    plot_chart(data_filtered, levels, current_price)

if __name__ == "__main__":
    main()