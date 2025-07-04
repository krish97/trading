import React, { useState, useEffect } from 'react'
import {
  Container,
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Divider,
  Paper,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Rating,
  Stack
} from '@mui/material'
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  SignalCellular4Bar,
  SignalCellularConnectedNoInternet4Bar,
  SignalCellular0Bar,
  ExpandMore,
  Psychology,
  Analytics,
  SentimentSatisfied,
  SentimentDissatisfied,
  SentimentNeutral
} from '@mui/icons-material'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api'

function App() {
  const [selectedStock, setSelectedStock] = useState('AAPL')
  const [signalData, setSignalData] = useState(null)
  const [priceData, setPriceData] = useState(null)
  const [sentimentData, setSentimentData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState(0)

  const stocks = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX', 'AMD', 'SNAP']

  useEffect(() => {
    const fetchData = async (stock) => {
      setLoading(true)
      setError(null)
      try {
        const [signalRes, priceRes, sentimentRes] = await Promise.all([
          axios.get(`${API_BASE}/signal/${stock}`),
          axios.get(`${API_BASE}/price/${stock}`),
          axios.get(`${API_BASE}/sentiment/${stock}`)
        ])

        if (signalRes.data.error || priceRes.data.error) {
          setError(signalRes.data.error || priceRes.data.error)
          setSignalData(null)
          setPriceData(null)
          setSentimentData(null)
        } else {
          setSignalData(signalRes.data)
          setPriceData(priceRes.data)
          setSentimentData(sentimentRes.data)
        }
      } catch (err) {
        setError(err.message || 'Failed to fetch data')
        setSignalData(null)
        setPriceData(null)
        setSentimentData(null)
      }
      setLoading(false)
    }

    if (selectedStock) {
      fetchData(selectedStock)
    }
  }, [selectedStock])

  const getSignalIcon = (signal) => {
    if (signal?.includes('BULLISH')) return <TrendingUp color="success" />
    if (signal?.includes('BEARISH')) return <TrendingDown color="error" />
    return <TrendingFlat color="warning" />
  }

  const getSignalColor = (signal) => {
    if (signal?.includes('BULLISH')) return 'success'
    if (signal?.includes('BEARISH')) return 'error'
    return 'warning'
  }

  const getConfidenceIcon = (confidence) => {
    if (confidence >= 80) return <SignalCellular4Bar color="success" />
    if (confidence >= 60) return <SignalCellularConnectedNoInternet4Bar color="warning" />
    return <SignalCellular0Bar color="error" />
  }

  const getSentimentIcon = (sentiment) => {
    if (sentiment >= 70) return <SentimentSatisfied color="success" />
    if (sentiment <= 30) return <SentimentDissatisfied color="error" />
    return <SentimentNeutral color="warning" />
  }

  const handleStockChange = (event) => {
    setSelectedStock(event.target.value)
  }

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue)
  }

  const renderTradingSignals = () => (
    <>
      {/* Signal Data */}
      {signalData && !loading && (
        <>
          {/* Main Signal Card */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  {getSignalIcon(signalData.signal)}
                  <Typography variant="h5" sx={{ ml: 1 }}>
                    {signalData.symbol}
                  </Typography>
                </Box>
                
                <Box display="flex" alignItems="center" mb={2}>
                  <Chip
                    label={signalData.signal}
                    color={getSignalColor(signalData.signal)}
                    size="large"
                    sx={{ mr: 2 }}
                  />
                  <Box display="flex" alignItems="center">
                    <Typography variant="body2" sx={{ ml: 0.5 }}>
                      Score: {signalData.score}
                    </Typography>
                  </Box>
                </Box>

                <Typography variant="h6" gutterBottom>
                  Signal Score: {signalData.score}
                </Typography>
                
                <LinearProgress
                  variant="determinate"
                  value={Math.abs(signalData.score) * 2}
                  color={getSignalColor(signalData.signal)}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </CardContent>
            </Card>
          </Grid>

          {/* Components Breakdown */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Indicator Components
                </Typography>
                <List dense>
                  {signalData.components && Object.entries(signalData.components).map(([key, component]) => (
                    <ListItem key={key}>
                      <ListItemText
                        primary={
                          <Box display="flex" justifyContent="space-between" alignItems="center">
                            <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                              {key.replace('_', ' ')}
                            </Typography>
                            <Chip
                              label={component.score > 0 ? 'Bullish' : component.score < 0 ? 'Bearish' : 'Neutral'}
                              color={component.score > 0 ? 'success' : component.score < 0 ? 'error' : 'warning'}
                              size="small"
                            />
                          </Box>
                        }
                        secondary={`Score: ${component.score.toFixed(2)}`}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>

          {/* Option Recommendations */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  💡 Option Trading Recommendations
                </Typography>
                <List>
                  {signalData.recommendations && signalData.recommendations.map((rec, index) => (
                    <React.Fragment key={index}>
                      <ListItem>
                        <ListItemText primary={rec} />
                      </ListItem>
                      {index < signalData.recommendations.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>
        </>
      )}
    </>
  )

  const renderSentimentAnalysis = () => (
    <>
      {sentimentData && !loading && (
        <>
          {/* Main Sentiment Overview */}
          <Grid item xs={12}>
            <Card sx={{ bgcolor: 'background.paper', border: 2, borderColor: 'primary.main' }}>
              <CardContent>
                <Box display="flex" alignItems="center" mb={3}>
                  <Psychology sx={{ mr: 2, fontSize: 40, color: 'primary.main' }} />
                  <Typography variant="h4" component="h2" sx={{ fontWeight: 'bold' }}>
                    Sentiment Analysis: {sentimentData.symbol}
                  </Typography>
                </Box>
                
                <Grid container spacing={3}>
                  <Grid item xs={12} md={4}>
                    <Box textAlign="center">
                      <Typography variant="h3" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                        {sentimentData.sentiment.overall_sentiment}%
                      </Typography>
                      <Typography variant="h6" gutterBottom>
                        Overall Sentiment
                      </Typography>
                      <Chip
                        label={sentimentData.sentiment.sentiment_label}
                        color={sentimentData.sentiment.sentiment_label === 'Bullish' ? 'success' : 'error'}
                        size="large"
                        icon={getSentimentIcon(sentimentData.sentiment.overall_sentiment)}
                      />
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Box textAlign="center">
                      <Typography variant="h3" sx={{ fontWeight: 'bold', color: 'secondary.main' }}>
                        {sentimentData.sentiment.confidence}%
                      </Typography>
                      <Typography variant="h6" gutterBottom>
                        Confidence Level
                      </Typography>
                      <Rating 
                        value={sentimentData.sentiment.confidence / 20} 
                        readOnly 
                        size="large"
                        precision={0.5}
                      />
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Box textAlign="center">
                      <Typography variant="h3" sx={{ fontWeight: 'bold', color: 'info.main' }}>
                        {sentimentData.sentiment.articles_analyzed}
                      </Typography>
                      <Typography variant="h6" gutterBottom>
                        Articles Analyzed
                      </Typography>
                      <Chip label="50 Articles" color="info" />
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Investment Recommendation */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  📊 Investment Recommendation
                </Typography>
                <Box display="flex" alignItems="center" mb={2}>
                  <Chip
                    label={sentimentData.sentiment.investment_recommendation}
                    color={sentimentData.sentiment.investment_recommendation === 'Buy' ? 'success' : 'error'}
                    size="large"
                  />
                </Box>
                <Typography variant="body2" color="text.secondary" paragraph>
                  {sentimentData.sentiment.intelligent_investor_summary}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Business Quality & Valuation */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  🏢 Business Assessment
                </Typography>
                <Stack spacing={2}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Business Quality
                    </Typography>
                    <Chip
                      label={sentimentData.sentiment.business_quality_assessment}
                      color={sentimentData.sentiment.business_quality_assessment === 'High' ? 'success' : 'warning'}
                    />
                  </Box>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Valuation Perspective
                    </Typography>
                    <Chip
                      label={sentimentData.sentiment.valuation_perspective}
                      color={sentimentData.sentiment.valuation_perspective.includes('Overvalued') ? 'error' : 'success'}
                    />
                  </Box>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Contrarian Signal
                    </Typography>
                    <Chip
                      label={sentimentData.sentiment.contrarian_signal}
                      color={sentimentData.sentiment.contrarian_signal === 'Avoid' ? 'error' : 'success'}
                    />
                  </Box>
                </Stack>
              </CardContent>
            </Card>
          </Grid>

          {/* Key Factors */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  📈 Key Bullish Factors
                </Typography>
                <List dense>
                  {sentimentData.sentiment.key_bullish_factors.map((factor, index) => (
                    <ListItem key={index}>
                      <ListItemText 
                        primary={factor}
                        primaryTypographyProps={{ variant: 'body2' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  📉 Key Bearish Factors
                </Typography>
                <List dense>
                  {sentimentData.sentiment.key_bearish_factors.map((factor, index) => (
                    <ListItem key={index}>
                      <ListItemText 
                        primary={factor}
                        primaryTypographyProps={{ variant: 'body2' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>

          {/* Summary Reasoning */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  🧠 AI Analysis Summary
                </Typography>
                <Typography variant="body1" paragraph>
                  {sentimentData.sentiment.summary_reasoning}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Analysis timestamp: {new Date(sentimentData.timestamp).toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </>
      )}
    </>
  )

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center" sx={{ fontWeight: 'bold' }}>
          📈 Advanced Trading System Dashboard
        </Typography>
        <Typography variant="h6" align="center" color="text.secondary" gutterBottom>
          Real-time sentiment analysis & technical indicators for intelligent trading decisions
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Stock Selector */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <FormControl fullWidth>
                <InputLabel>Select Stock</InputLabel>
                <Select
                  value={selectedStock}
                  label="Select Stock"
                  onChange={handleStockChange}
                >
                  {stocks.map((stock) => (
                    <MenuItem key={stock} value={stock}>{stock}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </CardContent>
          </Card>
        </Grid>

        {/* Current Stock Price Display */}
        {priceData && !loading && (
          <Grid item xs={12}>
            <Card sx={{ bgcolor: 'background.paper', border: 2, borderColor: 'primary.main' }}>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Box>
                    <Typography variant="h4" component="h2" gutterBottom sx={{ fontWeight: 'bold' }}>
                      {priceData.symbol}
                    </Typography>
                    <Typography variant="h3" component="h1" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                      ${priceData.price}
                    </Typography>
                    {priceData.price_source && (
                      <Chip 
                        label={priceData.price_source} 
                        color={priceData.price_source === 'After Hours' ? 'warning' : 'success'} 
                        size="small" 
                        sx={{ mt: 1 }}
                      />
                    )}
                  </Box>
                  <Box textAlign="right">
                    <Typography 
                      variant="h5" 
                      sx={{ 
                        color: priceData.change >= 0 ? 'success.main' : 'error.main',
                        fontWeight: 'bold'
                      }}
                    >
                      {priceData.change >= 0 ? '+' : ''}{priceData.change} ({priceData.change_percent >= 0 ? '+' : ''}{priceData.change_percent}%)
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Today's Change
                    </Typography>
                    {priceData.after_hours_price && priceData.regular_market_price && (
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        AH: ${priceData.after_hours_price} | RM: ${priceData.regular_market_price}
                      </Typography>
                    )}
                  </Box>
                </Box>
                <Box mt={2}>
                  <Typography variant="body2" color="text.secondary">
                    Trend: <Chip 
                      label={priceData.trend} 
                      color={priceData.trend === 'Bullish' ? 'success' : 'error'} 
                      size="small" 
                    />
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Tabs */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Tabs value={activeTab} onChange={handleTabChange} centered>
                <Tab 
                  icon={<Analytics />} 
                  label="Trading Signals" 
                  iconPosition="start"
                />
                <Tab 
                  icon={<Psychology />} 
                  label="Sentiment Analysis" 
                  iconPosition="start"
                />
              </Tabs>
            </CardContent>
          </Card>
        </Grid>

        {/* Error Display */}
        {error && (
          <Grid item xs={12}>
            <Alert severity="error">{error}</Alert>
          </Grid>
        )}

        {/* Loading State */}
        {loading && (
          <Grid item xs={12}>
            <Box display="flex" justifyContent="center" alignItems="center" p={3}>
              <CircularProgress />
              <Typography variant="body1" sx={{ ml: 2 }}>
                Fetching data...
              </Typography>
            </Box>
          </Grid>
        )}

        {/* Tab Content */}
        {activeTab === 0 && renderTradingSignals()}
        {activeTab === 1 && renderSentimentAnalysis()}

        {/* Summary */}
        {!loading && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2, bgcolor: 'background.paper' }}>
              <Typography variant="body2" color="text.secondary">
                Last updated: {new Date().toLocaleString()}
              </Typography>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Container>
  )
}

export default App 