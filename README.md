
# Smart Energy Consumption Optimizer ⚡

I built this project to track and predict household energy usage. The idea was to see if I could use Machine Learning to predict electricity bills based on things like temperature and time of day, and then find ways to reduce costs.

It’s a dashboard built with Streamlit that visualizes consumption data and gives recommendations on how to save energy (like shifting heavy usage to off-peak hours).

## What it does
* **Overview Tab:** Shows total cost, daily averages, and a breakdown of which appliances use the most power.
* **Predictions:** You can enter a date and temperature to see a predicted energy cost. I used a few ML models (saved as `.pkl` files) to handle the forecasting.
* **Anomalies:** Highlights weird spikes in usage that might indicate a leak or a device left running.
* **Calculator:** I added a custom calculator where you can input specific appliances (like your AC or gaming PC) to see exactly how much they contribute to the monthly bill.
* **Gamification:** Just a fun section to compare usage against a "neighborhood average."

## How to run it locally

1.  Clone this repo.
2.  Make sure you have the required libraries installed:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the app:
    ```bash
    streamlit run dashboard.py
    ```

## Files in this repo
* `dashboard.py`: The main code for the frontend and logic.
* `energy_consumption_data.csv`: The dataset I used for the visualizations.
* `models/`: This folder contains the pre-trained models (`consumption_model.pkl`, etc.). *Note: If the models aren't loading, the code has a fallback to use simple averages so the app doesn't crash.*

## To Do / Future updates
* Connect it to a real smart meter API (currently using CSV data).
* Improve the accuracy of the anomaly detection.
* Add a dark mode toggle.
