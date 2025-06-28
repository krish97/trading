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
  CircularProgress
} from '@mui/material'
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  SignalCellular4Bar,
  SignalCellularConnectedNoInternet4Bar,
  SignalCellular0Bar
} from '@mui/icons-material'
import axios from 'axios'

const API_BASE = 'http://localhost:8000'

function App() {
  const [selectedStock, setSelectedStock] = useState('AAPL')
  const [signalData, setSignalData] = useState(null)
  const [priceData, setPriceData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const stocks = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX', 'AMD', 'SNAP']

  const fetchSignal = async (symbol) => {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.get(`${API_BASE}/api/signal/${symbol}`)
      
      // Check if the response contains an error
      if (response.data.error) {
        setError(response.data.error)
        setSignalData(null)
        setPriceData(null)
        return
      }
      
      setSignalData(response.data)
      // Extract price data from the signal response
      setPriceData({
        symbol: response.data.symbol,
        price: response.data.current_price,
        after_hours_price: response.data.after_hours_price,
        change: 0, // We'll calculate this if needed
        change_percent: 0,
        price_source: response.data.after_hours_price ? 'After Hours' : 'Regular Market',
        regular_market_price: response.data.current_price,
        trend: response.data.signal.includes('BULLISH') ? 'Bullish' : response.data.signal.includes('BEARISH') ? 'Bearish' : 'Neutral'
      })
    } catch (err) {
      setError('Failed to fetch data. Make sure the backend is running.')
      setSignalData(null)
      setPriceData(null)
      console.error('Error fetching data:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchSignal(selectedStock)
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

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center" sx={{ fontWeight: 'bold' }}>
          📈 Trading System Dashboard
        </Typography>
        <Typography variant="h6" align="center" color="text.secondary" gutterBottom>
          Real-time sentiment analysis & technical indicators for option plays
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
                  onChange={(e) => setSelectedStock(e.target.value)}
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
                Fetching signal data...
              </Typography>
            </Box>
          </Grid>
        )}

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

            {/* Summary */}
            <Grid item xs={12}>
              <Paper sx={{ p: 2, bgcolor: 'background.paper' }}>
                <Typography variant="body2" color="text.secondary">
                  Last updated: {new Date(signalData.last_updated).toLocaleString()}
                </Typography>
              </Paper>
            </Grid>
          </>
        )}
      </Grid>
    </Container>
  )
}

export default App 