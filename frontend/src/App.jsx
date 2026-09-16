import {
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";

import "./App.css";

const API_BASE_URL =
  "http://127.0.0.1:8000";

function App() {

  // ============================================================
  // LIVE ETA
  // ============================================================

  const [
    trainNumber,
    setTrainNumber
  ] = useState("12919");

  const [
    etaData,
    setEtaData
  ] = useState(null);

  const [
    loading,
    setLoading
  ] = useState(false);

  const [
    error,
    setError
  ] = useState("");


  // ============================================================
  // REPLAY
  // ============================================================

  const [
    journeys,
    setJourneys
  ] = useState([]);

  const [
    replayTrain,
    setReplayTrain
  ] = useState("12919");

  const [
    replayDate,
    setReplayDate
  ] = useState("2026-09-10");

  const [
    replayData,
    setReplayData
  ] = useState(null);

  const [
    replayLoading,
    setReplayLoading
  ] = useState(false);

  const [
    replayError,
    setReplayError
  ] = useState("");

  const [
    currentStep,
    setCurrentStep
  ] = useState(0);

  const [
    isPlaying,
    setIsPlaying
  ] = useState(false);


  // ============================================================
  // CALENDAR STATE
  // ============================================================

  const [
    calendarMonth,
    setCalendarMonth
  ] = useState(
    new Date(2026, 8, 1)
  );


  // ============================================================
  // TIMELINE REFS
  // ============================================================

  const timelineWrapperRef =
    useRef(null);

  const timelineItemRefs =
    useRef([]);


  // ============================================================
  // LOAD HISTORICAL JOURNEYS
  // ============================================================

  useEffect(() => {
    loadJourneys();
  }, []);


  async function loadJourneys() {

    try {

      const response =
        await fetch(
          `${API_BASE_URL}/replay/journeys`
        );

      if (!response.ok) {
        throw new Error(
          "Unable to load historical journeys"
        );
      }

      const data =
        await response.json();

      if (data.success) {

        const loadedJourneys =
          data.journeys || [];

        setJourneys(
          loadedJourneys
        );

        // Automatically choose the first
        // available date for the default train
        // and open the calendar on that month.

        const firstJourney =
          loadedJourneys.find(
            (journey) =>
              String(
                journey.train_number
              ) === "12919"
          );

        if (firstJourney) {

          setReplayDate(
            firstJourney.journey_date
          );

          const firstDate =
            new Date(
              `${firstJourney.journey_date}T00:00:00`
            );

          setCalendarMonth(
            new Date(
              firstDate.getFullYear(),
              firstDate.getMonth(),
              1
            )
          );

        }

      }

    } catch (err) {

      console.error(err);

      setReplayError(
        "Unable to load historical journey list."
      );

    }

  }


  // ============================================================
  // LIVE ETA
  // ============================================================

  async function getETA() {

    const number =
      trainNumber.trim();

    setError("");
    setEtaData(null);

    if (!/^\d{5}$/.test(number)) {

      setError(
        "Please enter a valid 5-digit train number."
      );

      return;
    }

    setLoading(true);

    try {

      const response =
        await fetch(
          `${API_BASE_URL}/predict/${number}`
        );

      const data =
        await response.json();

      if (!response.ok) {

        throw new Error(
          data.detail ||
          "Unable to get train prediction."
        );

      }

      setEtaData(data);

    } catch (err) {

      console.error(err);

      setError(
        err.message ||
        "Failed to fetch ETA."
      );

    } finally {

      setLoading(false);

    }

  }


  // ============================================================
  // LOAD REPLAY
  // ============================================================

  async function loadReplay() {

    setReplayError("");
    setReplayData(null);
    setCurrentStep(0);
    setIsPlaying(false);

    if (!/^\d{5}$/.test(replayTrain)) {

      setReplayError(
        "Please select a valid 5-digit train number."
      );

      return;
    }

    if (!replayDate) {

      setReplayError(
        "Please select a journey date."
      );

      return;
    }

    setReplayLoading(true);

    try {

      const response =
        await fetch(
          `${API_BASE_URL}/replay/${replayTrain}/${replayDate}`
        );

      const data =
        await response.json();

      if (!response.ok) {

        throw new Error(
          data.detail ||
          "Unable to load replay."
        );

      }

      setReplayData(data);

    } catch (err) {

      console.error(err);

      setReplayError(
        err.message ||
        "Failed to load replay."
      );

    } finally {

      setReplayLoading(false);

    }

  }


  // ============================================================
  // REPLAY PLAYBACK
  // ============================================================

  useEffect(() => {

    if (
      !isPlaying ||
      !replayData
    ) {
      return;
    }

    if (
      currentStep >=
      replayData.steps.length - 1
    ) {

      setIsPlaying(false);

      return;
    }

    const timer =
      setTimeout(() => {

        setCurrentStep(
          (previous) =>
            previous + 1
        );

      }, 1200);

    return () =>
      clearTimeout(timer);

  }, [
    isPlaying,
    currentStep,
    replayData
  ]);


  // ============================================================
  // AUTO-SCROLL TO CURRENT STATION
  // ============================================================

  useEffect(() => {

    if (
      !replayData ||
      !timelineWrapperRef.current
    ) {
      return;
    }

    const activeStation =
      timelineItemRefs.current[
        currentStep
      ];

    if (!activeStation) {
      return;
    }

    const wrapper =
      timelineWrapperRef.current;

    const stationCenter =
      activeStation.offsetLeft +
      activeStation.offsetWidth / 2;

    const wrapperCenter =
      wrapper.clientWidth / 2;

    const targetScroll =
      stationCenter -
      wrapperCenter;

    const maxScroll =
      wrapper.scrollWidth -
      wrapper.clientWidth;

    const finalScroll =
      Math.max(
        0,
        Math.min(
          targetScroll,
          maxScroll
        )
      );

    wrapper.scrollTo({
      left: finalScroll,
      behavior: "smooth"
    });

  }, [
    currentStep,
    replayData
  ]);


  // ============================================================
  // REPLAY BUTTONS
  // ============================================================

  function toggleReplay() {

    if (!replayData) {
      return;
    }

    if (
      currentStep >=
      replayData.steps.length - 1
    ) {

      setCurrentStep(0);
      setIsPlaying(true);

      return;
    }

    setIsPlaying(
      (previous) =>
        !previous
    );

  }


  function resetReplay() {

    setCurrentStep(0);
    setIsPlaying(false);

  }


  // ============================================================
  // FORMAT DELAY
  // ============================================================

  function formatDelay(delay) {

    if (
      delay === null ||
      delay === undefined
    ) {
      return "N/A";
    }

    const rounded =
      Math.round(delay);

    if (rounded < 0) {

      return `${Math.abs(
        rounded
      )} min early`;

    }

    if (rounded === 0) {
      return "On time";
    }

    return `+${rounded} min`;

  }


  // ============================================================
  // FORMAT TIME
  // ============================================================

  function formatTime(value) {

    if (!value) {
      return "N/A";
    }

    try {

      const date =
        new Date(value);

      return date.toLocaleTimeString(
        "en-IN",
        {
          hour: "2-digit",
          minute: "2-digit",
          hour12: true
        }
      );

    } catch {

      return "N/A";

    }

  }


  // ============================================================
  // DELAY NUMBER
  // ============================================================

  function formatDelayChange(change) {

    if (
      change === null ||
      change === undefined ||
      Number.isNaN(Number(change))
    ) {
      return "N/A";
    }

    const value =
      Number(change);

    if (value > 0) {
      return `+${value.toFixed(1)} min`;
    }

    if (value < 0) {
      return `${value.toFixed(1)} min`;
    }

    return "0.0 min";

  }


  function getChangeClass(change) {

    if (
      change === null ||
      change === undefined ||
      Number.isNaN(Number(change))
    ) {
      return "";
    }

    if (Number(change) > 0) {
      return "danger-text";
    }

    if (Number(change) < 0) {
      return "success-text";
    }

    return "";

  }


  function getDelayNumber(delay) {

    if (
      delay === null ||
      delay === undefined
    ) {
      return 0;
    }

    return Math.round(delay);

  }


  // ============================================================
  // TRAIN LIST
  // ============================================================

  const uniqueTrains = [
    ...new Set(
      journeys.map(
        (journey) =>
          String(
            journey.train_number
          )
      )
    )
  ];


  // ============================================================
  // HISTORICAL CALENDAR
  // ============================================================

  const selectedTrainJourneys =
    useMemo(() => {
      return journeys.filter(
        (journey) =>
          String(
            journey.train_number
          ) ===
          String(replayTrain)
      );
    }, [
      journeys,
      replayTrain
    ]);


  const operatedDates =
    useMemo(() => {
      return new Set(
        selectedTrainJourneys.map(
          (journey) =>
            journey.journey_date
        )
      );
    }, [
      selectedTrainJourneys
    ]);


  function getDateKey(date) {

    const year =
      date.getFullYear();

    const month =
      String(
        date.getMonth() + 1
      ).padStart(2, "0");

    const day =
      String(
        date.getDate()
      ).padStart(2, "0");

    return `${year}-${month}-${day}`;

  }


  function getCalendarDays() {

    const year =
      calendarMonth.getFullYear();

    const month =
      calendarMonth.getMonth();

    const firstDay =
      new Date(
        year,
        month,
        1
      );

    const lastDay =
      new Date(
        year,
        month + 1,
        0
      );

    // Convert Sunday-first JS index
    // into Monday-first calendar layout.

    const startOffset =
      (
        firstDay.getDay() +
        6
      ) % 7;

    const daysInMonth =
      lastDay.getDate();

    const cells = [];

    for (
      let i = 0;
      i < startOffset;
      i++
    ) {
      cells.push(null);
    }

    for (
      let day = 1;
      day <= daysInMonth;
      day++
    ) {
      cells.push(
        new Date(
          year,
          month,
          day
        )
      );
    }

    return cells;

  }


  function goToPreviousMonth() {

    setCalendarMonth(
      (previous) =>
        new Date(
          previous.getFullYear(),
          previous.getMonth() - 1,
          1
        )
    );

  }


  function goToNextMonth() {

    setCalendarMonth(
      (previous) =>
        new Date(
          previous.getFullYear(),
          previous.getMonth() + 1,
          1
        )
    );

  }


  function selectCalendarDate(date) {

    if (!date) {
      return;
    }

    const dateKey =
      getDateKey(date);

    if (
      !operatedDates.has(
        dateKey
      )
    ) {
      return;
    }

    setReplayDate(
      dateKey
    );

    setReplayError("");

    loadReplay(
      dateKey,
      replayTrain
    );

  }


  function isSelectedDate(date) {

    if (
      !date ||
      !replayDate
    ) {
      return false;
    }

    return (
      getDateKey(date) ===
      replayDate
    );

  }


  function formatCalendarMonth() {

    return calendarMonth.toLocaleDateString(
      "en-IN",
      {
        month: "long",
        year: "numeric"
      }
    );

  }


  // ============================================================
  // CURRENT REPLAY STEP
  // ============================================================

  const replayStep =
    replayData?.steps?.[
      currentStep
    ] || null;


  // ============================================================
  // TIMELINE PROGRESS
  // ============================================================

  const timelineProgress =
    replayData &&
    replayData.steps.length > 1
      ? (
          currentStep /
          (
            replayData.steps.length - 1
          )
        ) * 100
      : 0;


  // ============================================================
  // UI
  // ============================================================

  return (

    <div className="app">


      {/* ======================================================
          SIDEBAR
      ====================================================== */}

      <aside className="sidebar">

        <div className="brand">

          <div className="brand-icon">
            🚆
          </div>

          <div>

            <h1>
              TrainETA
            </h1>

            <span>
              Smarter Journeys, Together
            </span>

          </div>

        </div>


        <nav className="sidebar-nav">

          <a
            href="#dashboard"
            className="nav-item active"
          >
            <span>⌂</span>
            Dashboard
          </a>

          <a
            href="#live"
            className="nav-item"
          >
            <span>⌕</span>
            Search Train
          </a>

          <a
            href="#live-status"
            className="nav-item"
          >
            <span>◉</span>
            Live Status
          </a>

          <a
            href="#route"
            className="nav-item"
          >
            <span>⌁</span>
            Train Route
          </a>

          <a
            href="#prediction"
            className="nav-item"
          >
            <span>◷</span>
            Predicted ETA
          </a>

          <a
            href="#analytics"
            className="nav-item"
          >
            <span>▥</span>
            Analytics
          </a>

          <a
            href="#about"
            className="nav-item"
          >
            <span>ⓘ</span>
            About
          </a>

        </nav>


        <div className="sidebar-bottom">

          <div className="rail-decoration">
            🚆
          </div>

          <p>
            Bharat ki Rail
            <br />
            Bharat ka Vikas
          </p>

        </div>

      </aside>


      {/* ======================================================
          MAIN AREA
      ====================================================== */}

      <div className="main-area">


        {/* HEADER */}

        <header className="top-header">

          <div>

            <h2>
              Real-Time Train Tracking & ETA Prediction
            </h2>

            <p>
              AI-powered insights for a smoother journey
            </p>

          </div>


          <div className="header-right">

            <div className="live-status-pill">

              <span className="live-dot"></span>

              Live Updates

            </div>


            <div className="railway-badge">

              <span className="railway-emblem">
                🔴
              </span>

              <div>

                <strong>
                  Indian Railways
                </strong>

                <small>
                  Nation's Lifeline
                </small>

              </div>

            </div>

          </div>

        </header>


        {/* ====================================================
            CONTENT
        ==================================================== */}

        <main
          className="content"
          id="dashboard"
        >


          {/* ==================================================
              SEARCH
          ================================================== */}

          <section
            className="search-section"
            id="live"
          >

            <div className="section-title">

              <h1>
                Track Your Train
              </h1>

              <p>
                Enter train number to get real-time
                status and ETA predictions
              </p>

            </div>


            <div className="search-box">

              <div className="search-input-wrapper">

                <span className="search-icon">
                  🔍
                </span>

                <input
                  type="text"
                  value={trainNumber}
                  onChange={(event) =>
                    setTrainNumber(
                      event.target.value.replace(
                        /\D/g,
                        ""
                      )
                    )
                  }
                  onKeyDown={(event) => {

                    if (
                      event.key === "Enter"
                    ) {
                      getETA();
                    }

                  }}
                  maxLength={5}
                  placeholder="Enter 5-digit train number"
                />

                {trainNumber && (

                  <button
                    className="clear-search"
                    onClick={() =>
                      setTrainNumber("")
                    }
                  >
                    ×
                  </button>

                )}

              </div>


              <button
                className="search-button"
                onClick={getETA}
                disabled={loading}
              >

                {loading
                  ? "Fetching..."
                  : "Search"}

              </button>

            </div>


            <p className="search-hint">
              Try: 12919, 12951, 12952, 12002
            </p>


            {error && (

              <div className="error-box">
                ⚠️ {error}
              </div>

            )}

          </section>


          {/* ==================================================
              LIVE RESULTS
          ================================================== */}

          {etaData && (

            <section
              className="results"
              id="live-status"
            >


              {/* TRAIN SUMMARY */}

              <div className="train-summary-card">

                <div className="train-summary-left">

                  <div className="train-logo">
                    🚆
                  </div>

                  <div>

                    <div className="train-id-row">

                      <span className="train-number">
                        {etaData.train_number}
                      </span>

                      <span className="running-badge">

                        <span></span>

                        Running

                      </span>

                    </div>

                    <h2>
                      {etaData.train_name}
                    </h2>

                    <p>
                      Journey Date:{" "}
                      {etaData.journey_date}
                    </p>

                  </div>

                </div>


                <div className="eta-highlight">

                  <span>
                    Expected Arrival
                  </span>

                  <strong>
                    {formatTime(
                      etaData.expected_arrival
                    )}
                  </strong>

                  <small>
                    at{" "}
                    {etaData.next_station}
                  </small>

                </div>

              </div>


              {/* STATS */}

              <div className="stats-grid">

                <div className="stat-card">

                  <div className="stat-icon blue">
                    📍
                  </div>

                  <div>

                    <span>
                      Current Station
                    </span>

                    <strong>
                      {etaData.current_station}
                    </strong>

                    <small>
                      {etaData.current_station_name}
                    </small>

                  </div>

                </div>


                <div className="stat-card">

                  <div className="stat-icon purple">
                    ◉
                  </div>

                  <div>

                    <span>
                      Current Delay
                    </span>

                    <strong
                      className={
                        getDelayNumber(
                          etaData.current_delay_minutes
                        ) > 15
                          ? "danger-text"
                          : "success-text"
                      }
                    >
                      {formatDelay(
                        etaData.current_delay_minutes
                      )}
                    </strong>

                    <small>
                      vs. schedule
                    </small>

                  </div>

                </div>


                <div className="stat-card">

                  <div className="stat-icon orange">
                    🤖
                  </div>

                  <div>

                    <span>
                      AI Predicted Delay
                    </span>

                    <strong>
                      {formatDelay(
                        etaData.predicted_delay_minutes
                      )}
                    </strong>

                    <small>
                      at next station
                    </small>

                  </div>

                </div>


                <div className="stat-card">

                  <div className="stat-icon green">
                    ➜
                  </div>

                  <div>

                    <span>
                      Next Station
                    </span>

                    <strong>
                      {etaData.next_station}
                    </strong>

                    <small>
                      {etaData.next_station_name}
                    </small>

                  </div>

                </div>

              </div>


              {/* DYNAMIC FORECAST */}

              <div className="dashboard-card">

                <div className="card-heading">

                  <div>

                    <h3>
                      Dynamic Delay Forecast
                    </h3>

                    <p>
                      How the ML model adjusts the current delay
                    </p>

                  </div>

                  <span className="ai-badge">
                    ✦ AI MODEL
                  </span>

                </div>


                <div className="comparison-box">

                  <div>

                    <span>
                      Current Delay
                    </span>

                    <strong>
                      {formatDelay(
                        etaData.current_delay_minutes
                      )}
                    </strong>

                  </div>


                  <div className="comparison-arrow">
                    +
                  </div>


                  <div>

                    <span>
                      Predicted Change
                    </span>

                    <strong
                      className={getChangeClass(
                        etaData.predicted_delay_change_minutes
                      )}
                    >
                      {formatDelayChange(
                        etaData.predicted_delay_change_minutes
                      )}
                    </strong>

                  </div>


                  <div className="comparison-arrow">
                    =
                  </div>


                  <div>

                    <span>
                      Predicted Next Delay
                    </span>

                    <strong className="prediction-time">
                      {formatDelay(
                        etaData.predicted_delay_minutes
                      )}
                    </strong>

                  </div>

                </div>


                <div className="info-card">

                  <div className="info-icon">
                    💡
                  </div>

                  <div>

                    <h3>
                      What the model is doing
                    </h3>

                    <p>
                      The model starts with the train's
                      current observed delay and predicts
                      how that delay is expected to change
                      before the next station.
                    </p>

                  </div>

                </div>

              </div>


              {/* ROUTE */}

              <div
                className="dashboard-card route-progress-card"
                id="route"
              >

                <div className="card-heading">

                  <div>

                    <h3>
                      Route Progress
                    </h3>

                    <p>
                      Current position and next station
                    </p>

                  </div>

                  <span className="live-route-label">
                    ● LIVE
                  </span>

                </div>


                <div className="journey-route">

                  <div className="route-station-card current-station-card">

                    <div className="route-station-top">

                      <span className="station-status current-status">
                        CURRENT
                      </span>

                      <span className="station-icon">
                        📍
                      </span>

                    </div>

                    <strong>
                      {etaData.current_station}
                    </strong>

                    <p>
                      {etaData.current_station_name}
                    </p>

                    <span className="station-delay">
                      {formatDelay(
                        etaData.current_delay_minutes
                      )}
                    </span>

                  </div>


                  <div className="route-connection">

                    <div className="connection-label">
                      Train moving toward next station
                    </div>

                    <div className="connection-line">

                      <div className="connection-fill"></div>

                      <div className="moving-train">
                        🚆
                      </div>

                    </div>

                    <div className="connection-info">

                      <span>
                        Current:{" "}
                        <strong>
                          {formatDelay(
                            etaData.current_delay_minutes
                          )}
                        </strong>
                      </span>

                      <span>
                        Predicted:{" "}
                        <strong>
                          {formatDelay(
                            etaData.predicted_delay_minutes
                          )}
                        </strong>
                      </span>

                    </div>

                  </div>


                  <div className="route-station-card next-station-card">

                    <div className="route-station-top">

                      <span className="station-status next-status">
                        NEXT STATION
                      </span>

                      <span className="station-icon">
                        ○
                      </span>

                    </div>

                    <strong>
                      {etaData.next_station}
                    </strong>

                    <p>
                      {etaData.next_station_name}
                    </p>

                    <span className="station-eta">
                      ETA{" "}
                      {formatTime(
                        etaData.expected_arrival
                      )}
                    </span>

                  </div>

                </div>


                <div className="route-summary">

                  <div>

                    <span>
                      Current Location
                    </span>

                    <strong>
                      {etaData.current_station}
                    </strong>

                  </div>

                  <div className="route-summary-arrow">
                    →
                  </div>

                  <div>

                    <span>
                      Next Station
                    </span>

                    <strong>
                      {etaData.next_station}
                    </strong>

                  </div>

                  <div className="route-summary-divider"></div>

                  <div>

                    <span>
                      Expected Arrival
                    </span>

                    <strong className="route-summary-eta">
                      {formatTime(
                        etaData.expected_arrival
                      )}
                    </strong>

                  </div>

                </div>

              </div>


              {/* TWO COLUMN */}

              <div className="two-column">

                <div
                  className="dashboard-card prediction-main-card"
                  id="prediction"
                >

                  <div className="card-heading">

                    <div>

                      <h3>
                        AI-Powered ETA Prediction
                      </h3>

                      <p>
                        Dynamic machine learning forecast
                      </p>

                    </div>

                    <span className="ai-badge">
                      ✦ AI MODEL
                    </span>

                  </div>


                  <div className="prediction-main">

                    <div className="prediction-clock">
                      🕐
                    </div>

                    <div>

                      <span className="prediction-label">
                        EXPECTED ARRIVAL
                      </span>

                      <strong>
                        {formatTime(
                          etaData.expected_arrival
                        )}
                      </strong>

                      <p>
                        Expected delay:{" "}
                        <b>
                          {formatDelay(
                            etaData.predicted_delay_minutes
                          )}
                        </b>
                      </p>

                    </div>

                  </div>


                  <div className="comparison-box">

                    <div>

                      <span>
                        Scheduled
                      </span>

                      <strong>
                        {formatTime(
                          etaData.scheduled_arrival
                        )}
                      </strong>

                    </div>

                    <div className="comparison-arrow">
                      →
                    </div>

                    <div>

                      <span>
                        ML Prediction
                      </span>

                      <strong className="prediction-time">
                        {formatTime(
                          etaData.expected_arrival
                        )}
                      </strong>

                    </div>

                  </div>

                </div>


                <div className="dashboard-card details-card">

                  <div className="card-heading">

                    <div>

                      <h3>
                        Prediction Details
                      </h3>

                      <p>
                        Factors used by the model
                      </p>

                    </div>

                  </div>


                  <div className="detail-list">

                    <div className="detail-row">

                      <span>
                        ML Model
                      </span>

                      <strong>
                        {etaData.model}
                      </strong>

                    </div>


                    <div className="detail-row">

                      <span>
                        Current Delay
                      </span>

                      <strong>
                        {formatDelay(
                          etaData.current_delay_minutes
                        )}
                      </strong>

                    </div>


                    <div className="detail-row">

                      <span>
                        Previous Delay
                      </span>

                      <strong>
                        {formatDelay(
                          etaData.previous_delay_minutes
                        )}
                      </strong>

                    </div>


                    <div className="detail-row">

                      <span>
                        Predicted Change
                      </span>

                      <strong
                        className={getChangeClass(
                          etaData.predicted_delay_change_minutes
                        )}
                      >
                        {formatDelayChange(
                          etaData.predicted_delay_change_minutes
                        )}
                      </strong>

                    </div>


                    <div className="detail-row">

                      <span>
                        Scheduled Travel
                      </span>

                      <strong>
                        {etaData.scheduled_travel_time_minutes ?? "N/A"}
                        {" "}
                        min
                      </strong>

                    </div>


                    <div className="detail-row">

                      <span>
                        Next Station
                      </span>

                      <strong>
                        {etaData.next_station}
                      </strong>

                    </div>

                  </div>


                  <div className="confidence-box">

                    <div className="confidence-icon">
                      📊
                    </div>

                    <div>

                      <strong>
                        ML Prediction Active
                      </strong>

                      <span>
                        Based on historical train
                        behavior and live data
                      </span>

                    </div>

                  </div>

                </div>

              </div>


              {/* INFORMATION */}

              <div
                className="info-card"
                id="analytics"
              >

                <div className="info-icon">
                  💡
                </div>

                <div>

                  <h3>
                    How the prediction works
                  </h3>

                  <p>
                    Our machine learning model considers
                    the train's current delay, previous
                    delay, station position, scheduled
                    travel time and route information
                    to estimate how the delay will change
                    before the next station.
                  </p>

                </div>

              </div>

            </section>

          )}


          {/* ==================================================
              HISTORICAL REPLAY
          ================================================== */}

          <section
            className="replay-section"
            id="about"
          >

            <div className="replay-section-header">

              <div>

                <div className="eyebrow">
                  HISTORICAL ANALYSIS
                </div>

                <h2>
                  Historical Journey Replay
                </h2>

                <p>
                  Explore recorded train journeys using
                  the calendar and station-by-station replay.
                </p>

              </div>

              <div className="historical-badge">
                🕐 Historical Data
              </div>

            </div>


            {/* REPLAY SELECTOR */}

            <div className="replay-selector">

              <div className="selector-field">

                <label>
                  Train Number
                </label>

                <select
                  value={replayTrain}
                  onChange={(event) => {

                    const value =
                      event.target.value;

                    setReplayTrain(value);

                    const firstJourney =
                      journeys.find(
                        (journey) =>
                          String(
                            journey.train_number
                          ) === value
                      );

                    if (firstJourney) {

                      const firstDate =
                        firstJourney.journey_date;

                      setReplayDate(
                        firstDate
                      );

                      const firstDateObject =
                        new Date(
                          `${firstDate}T00:00:00`
                        );

                      setCalendarMonth(
                        new Date(
                          firstDateObject.getFullYear(),
                          firstDateObject.getMonth(),
                          1
                        )
                      );

                    } else {

                      setReplayDate("");

                    }

                    setReplayData(null);
                    setCurrentStep(0);
                    setIsPlaying(false);

                  }}
                >

                  {uniqueTrains.map(
                    (number) => (

                      <option
                        key={number}
                        value={number}
                      >
                        {number}
                      </option>

                    )
                  )}

                </select>

              </div>


              <div className="selector-field">

                <label>
                  Journey Date
                </label>

                <select
                  value={replayDate}
                  onChange={(event) => {

                    setReplayDate(
                      event.target.value
                    );

                    setReplayData(null);
                    setCurrentStep(0);
                    setIsPlaying(false);

                  }}
                >

                  {selectedTrainJourneys.map(
                    (journey) => (

                      <option
                        key={
                          journey.journey_date
                        }
                        value={
                          journey.journey_date
                        }
                      >
                        {journey.journey_date}
                      </option>

                    )
                  )}

                </select>

              </div>


              <button
                className="replay-load-button"
                onClick={loadReplay}
                disabled={replayLoading}
              >

                {replayLoading
                  ? "Loading..."
                  : "Load Replay →"}

              </button>

            </div>


            {/* ==================================================
                SMART HISTORICAL RUNNING CALENDAR
            ================================================== */}

            <div className="dashboard-card historical-calendar-card">

              <div className="card-heading">

                <div>

                  <h3>
                    Smart Historical Running Calendar
                  </h3>

                  <p>
                    Select a green date to explore the
                    recorded station-by-station journey.
                  </p>

                </div>


                <span className="historical-count">
                  {selectedTrainJourneys.length}
                  {" "}
                  recorded journeys
                </span>

              </div>


              <div className="calendar-header">

                <button
                  className="calendar-nav-button"
                  onClick={
                    goToPreviousMonth
                  }
                  aria-label="Previous month"
                >
                  ‹
                </button>


                <h3>
                  {formatCalendarMonth()}
                </h3>


                <button
                  className="calendar-nav-button"
                  onClick={
                    goToNextMonth
                  }
                  aria-label="Next month"
                >
                  ›
                </button>

              </div>


              <div className="calendar-legend">

                <div className="legend-item">

                  <span className="legend-dot operated"></span>

                  <span>
                    Historical journey available
                  </span>

                </div>


                <div className="legend-item">

                  <span className="legend-dot no-record"></span>

                  <span>
                    No historical record
                  </span>

                </div>

              </div>


              <div className="calendar-grid">

                {[
                  "Mon",
                  "Tue",
                  "Wed",
                  "Thu",
                  "Fri",
                  "Sat",
                  "Sun"
                ].map(
                  (day) => (

                    <div
                      key={day}
                      className="calendar-weekday"
                    >
                      {day}
                    </div>

                  )
                )}


                {getCalendarDays().map(
                  (date, index) => {

                    if (!date) {

                      return (
                        <div
                          key={`empty-${index}`}
                          className="calendar-day empty"
                        />
                      );

                    }

                    const dateKey =
                      getDateKey(date);

                    const operated =
                      operatedDates.has(
                        dateKey
                      );

                    const selected =
                      isSelectedDate(
                        date
                      );

                    return (

                      <button
                        key={dateKey}
                        className={
                          `calendar-day ${
                            operated
                              ? "operated"
                              : "no-record"
                          } ${
                            selected
                              ? "selected"
                              : ""
                          }`
                        }
                        onClick={() =>
                          selectCalendarDate(
                            date
                          )
                        }
                        disabled={
                          !operated
                        }
                        title={
                          operated
                            ? "Historical journey available — click to replay"
                            : "No historical journey record"
                        }
                      >

                        <span className="calendar-day-number">
                          {date.getDate()}
                        </span>

                        {operated && (

                          <span className="calendar-day-dot">
                            ●
                          </span>

                        )}

                      </button>

                    );

                  }
                )}

              </div>


              <div className="calendar-note">

                <span>
                  💡
                </span>

                <p>
                  A green date means the current dataset
                  contains a recorded journey for this
                  train on that date. A grey date means
                  there is no historical record available
                  in the dataset. It is not treated as a
                  cancellation because cancellation data is
                  not available from this endpoint.
                </p>

              </div>

            </div>


            {replayError && (

              <div className="error-box">
                ⚠️ {replayError}
              </div>

            )}


            {/* REPLAY LOADING */}

            {replayLoading && (

              <div className="dashboard-card replay-loading">

                <div className="loading-spinner">
                  ⟳
                </div>

                <strong>
                  Loading historical journey...
                </strong>

                <span>
                  Fetching station-by-station records.
                </span>

              </div>

            )}


            {/* REPLAY RESULT */}

            {replayData && replayStep && (

              <div className="replay-result">


                {/* HEADER */}

                <div className="replay-top-card">

                  <div>

                    <span className="eyebrow">
                      TRAIN JOURNEY
                    </span>

                    <h2>
                      🚆 Train{" "}
                      {replayData.train_number}
                    </h2>

                    <p>
                      Historical journey •{" "}
                      {replayData.journey_date}
                    </p>

                  </div>


                  <div className="replay-counter">

                    <strong>
                      {currentStep + 1}
                    </strong>

                    <span>
                      /{" "}
                      {replayData.steps.length}
                    </span>

                  </div>

                </div>


                {/* CURRENT STATION */}

                <div className="replay-current-card">

                  <div className="current-station-icon">
                    🚆
                  </div>

                  <div>

                    <span>
                      CURRENT REPLAY POSITION
                    </span>

                    <strong>
                      {replayStep.station_code}
                    </strong>

                    <h2>
                      {replayStep.station_name}
                    </h2>

                    <p>
                      Station sequence:{" "}
                      {replayStep.station_sequence}
                    </p>

                  </div>


                  <div className="replay-delay-big">

                    <span>
                      Arrival Delay
                    </span>

                    <strong>
                      {formatDelay(
                        replayStep.arrival_delay
                      )}
                    </strong>

                  </div>

                </div>


                {/* METRICS */}

                <div className="replay-metrics">

                  <div>

                    <span>
                      Scheduled Arrival
                    </span>

                    <strong>
                      {formatTime(
                        replayStep.scheduled_arrival
                      )}
                    </strong>

                  </div>


                  <div>

                    <span>
                      Actual Arrival
                    </span>

                    <strong>
                      {formatTime(
                        replayStep.actual_arrival
                      )}
                    </strong>

                  </div>


                  <div>

                    <span>
                      Arrival Delay
                    </span>

                    <strong>
                      {formatDelay(
                        replayStep.arrival_delay
                      )}
                    </strong>

                  </div>


                  <div>

                    <span>
                      Departure Delay
                    </span>

                    <strong>
                      {formatDelay(
                        replayStep.departure_delay
                      )}
                    </strong>

                  </div>

                </div>


                {/* PROGRESS */}

                <div className="replay-progress">

                  <div className="progress-heading">

                    <span>
                      Journey Progress
                    </span>

                    <strong>
                      {Math.round(
                        (
                          (currentStep + 1) /
                          replayData.steps.length
                        ) * 100
                      )}%
                    </strong>

                  </div>


                  <div className="progress-track">

                    <div
                      className="progress-fill"
                      style={{
                        width: `${
                          (
                            (currentStep + 1) /
                            replayData.steps.length
                          ) * 100
                        }%`
                      }}
                    />

                  </div>

                </div>


                {/* BUTTONS */}

                <div className="replay-buttons">

                  <button
                    onClick={toggleReplay}
                    className="play-button"
                  >

                    {isPlaying
                      ? "⏸ Pause Replay"
                      : "▶ Play Replay"}

                  </button>


                  <button
                    onClick={resetReplay}
                    className="reset-button"
                  >
                    ↺ Reset
                  </button>

                </div>


                {/* ==================================================
                    NEW RAILWAY TIMELINE
                ================================================== */}

                <div className="timeline-card">

                  <div className="timeline-heading">

                    <div>

                      <h3>
                        Journey Timeline
                      </h3>

                      <p>
                        Watch the train progress through
                        each station. Click any station to
                        jump to that point.
                      </p>

                    </div>


                    <span>
                      {replayData.steps.length}
                      {" "}
                      stations
                    </span>

                  </div>


                  {/* SCROLL CONTAINER */}

                  <div
                    className="timeline-wrapper"
                    ref={timelineWrapperRef}
                  >

                    <div
                      className="timeline"
                      style={{
                        "--timeline-progress":
                          timelineProgress
                      }}
                    >


                      {/* =================================================
                          RAILWAY TRACK
                      ================================================= */}

                      <div className="timeline-track">

                        <div className="timeline-track-base"></div>

                        <div className="timeline-track-progress"></div>

                      </div>


                      {/* =================================================
                          STATIONS
                      ================================================= */}

                      <div className="timeline-stations">

                        {replayData.steps.map(
                          (step, index) => {

                            const isCurrent =
                              index ===
                              currentStep;

                            const isCompleted =
                              index <
                              currentStep;

                            return (

                              <div
                                key={index}
                                className="timeline-station-column"
                              >


                                {/* STATION MARKER */}

                                <button
                                  ref={(element) => {

                                    timelineItemRefs.current[
                                      index
                                    ] = element;

                                  }}
                                  className={
                                    `timeline-marker-button ${
                                      isCurrent
                                        ? "current"
                                        : ""
                                    } ${
                                      isCompleted
                                        ? "completed"
                                        : ""
                                    }`
                                  }
                                  onClick={() => {

                                    setCurrentStep(
                                      index
                                    );

                                    setIsPlaying(
                                      false
                                    );

                                  }}
                                  aria-label={
                                    `Go to station ${
                                      step.station_code
                                    }`
                                  }
                                >

                                  <span className="timeline-marker">

                                    {isCompleted
                                      ? "✓"
                                      : index + 1}

                                  </span>

                                </button>


                                {/* STATION CARD */}

                                <button
                                  className={
                                    `timeline-station-card ${
                                      isCurrent
                                        ? "current"
                                        : ""
                                    } ${
                                      isCompleted
                                        ? "completed"
                                        : ""
                                    }`
                                  }
                                  onClick={() => {

                                    setCurrentStep(
                                      index
                                    );

                                    setIsPlaying(
                                      false
                                    );

                                  }}
                                >

                                  <div className="timeline-card-number">

                                    {isCompleted
                                      ? "✓"
                                      : index + 1}

                                  </div>


                                  <strong>
                                    {step.station_code}
                                  </strong>


                                  <span className="timeline-station-name">
                                    {step.station_name}
                                  </span>


                                  <span
                                    className={
                                      `timeline-delay ${
                                        getDelayNumber(
                                          step.arrival_delay
                                        ) > 0
                                          ? "late"
                                          : "ontime"
                                      }`
                                    }
                                  >

                                    {formatDelay(
                                      step.arrival_delay
                                    )}

                                  </span>


                                  <span className="timeline-time">

                                    Arr{" "}
                                    {formatTime(
                                      step.actual_arrival
                                    )}

                                  </span>


                                  <span className="timeline-time">

                                    Dep{" "}
                                    {formatTime(
                                      step.actual_departure
                                    )}

                                  </span>

                                </button>

                              </div>

                            );

                          }
                        )}

                      </div>


                      {/* =================================================
                          MOVING TRAIN
                      ================================================= */}

                      <div
                        className="timeline-moving-train"
                        style={{
                          left: `${
                            replayData.steps.length > 1
                              ? (
                                  currentStep /
                                  (
                                    replayData.steps.length -
                                    1
                                  )
                                ) * 100
                              : 0
                          }%`
                        }}
                      >

                        <div className="current-station-label">
                          Current Station
                        </div>

                        <div className="train-glow"></div>

                        <div className="train-icon">
                          🚆
                        </div>

                      </div>

                    </div>

                  </div>


                  {/* TIMELINE LEGEND */}

                  <div className="timeline-legend">

                    <div>

                      <span className="legend-line completed-line"></span>

                      Journey completed

                    </div>


                    <div>

                      <span className="legend-line upcoming-line"></span>

                      Upcoming stations

                    </div>


                    <div>

                      <span className="legend-train">
                        🚆
                      </span>

                      Current position

                    </div>

                  </div>

                </div>

              </div>

            )}

          </section>

        </main>


        {/* ======================================================
            FOOTER
        ====================================================== */}

        <footer>

          <div>
            🚆 <strong>TrainETA</strong>
          </div>

          <span>
            AI-powered railway intelligence
          </span>

          <span>
            •
          </span>

          <span>
            Smart India Hackathon 2026
          </span>

        </footer>

      </div>

    </div>

  );

}

export default App;