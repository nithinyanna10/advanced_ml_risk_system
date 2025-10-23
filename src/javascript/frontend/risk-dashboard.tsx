/**
 * Advanced React Risk Dashboard
 * Real-time financial risk assessment dashboard
 * 
 * Technologies: React, TypeScript, D3.js, WebSocket, Material-UI
 * Author: Nithin Yanna
 * Date: 2025
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  Alert,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  InputLabel,
  FormControl,
  TextField,
  Tabs,
  Tab,
  Badge,
  Avatar,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  CircularProgress,
  Snackbar,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Warning,
  CheckCircle,
  Error,
  Info,
  Refresh,
  Settings,
  Notifications,
  Security,
  Assessment,
  Timeline,
  BarChart,
  PieChart,
  TableChart,
  FilterList,
  Download,
  Share,
  Fullscreen,
  Close,
  PlayArrow,
  Pause,
  Stop,
  Speed,
  Memory,
  Storage,
  NetworkCheck,
  CloudDone,
  CloudOff,
  Sync,
  SyncProblem,
} from '@mui/icons-material';
import * as d3 from 'd3';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart as RechartsBarChart, Bar, PieChart as RechartsPieChart, Cell, AreaChart, Area } from 'recharts';
import { format, parseISO, subMinutes, subHours, subDays } from 'date-fns';

// Types
interface RiskAssessment {
  id: string;
  transactionId: string;
  customerId: string;
  amount: number;
  riskScore: number;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  confidence: number;
  recommendedAction: 'APPROVE' | 'REVIEW' | 'BLOCK' | 'ESCALATE';
  timestamp: string;
  processingTime: number;
  features: Record<string, number>;
  explanation?: RiskExplanation;
  businessRules: string[];
}

interface RiskExplanation {
  featureImportance: Record<string, number>;
  shapValues: Record<string, number>;
  limeExplanation: Array<{ feature: string; score: number }>;
  decisionPath: string[];
}

interface RiskStatistics {
  totalAssessments: number;
  highRiskCount: number;
  mediumRiskCount: number;
  lowRiskCount: number;
  averageRiskScore: number;
  averageProcessingTime: number;
  successRate: number;
  errorRate: number;
}

interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  networkLatency: number;
  activeConnections: number;
  requestsPerSecond: number;
  errorRate: number;
  cacheHitRate: number;
}

interface ModelPerformance {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  confusionMatrix: number[][];
  featureImportance: Record<string, number>;
  driftScore: number;
  lastUpdated: string;
}

// WebSocket hook
const useWebSocket = (url: string) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');

  useEffect(() => {
    const ws = new WebSocket(url);
    
    ws.onopen = () => {
      setConnectionStatus('connected');
      setSocket(ws);
    };
    
    ws.onmessage = (event) => {
      setLastMessage(JSON.parse(event.data));
    };
    
    ws.onclose = () => {
      setConnectionStatus('disconnected');
      setSocket(null);
    };
    
    ws.onerror = () => {
      setConnectionStatus('disconnected');
    };

    return () => {
      ws.close();
    };
  }, [url]);

  const sendMessage = useCallback((message: any) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    }
  }, [socket]);

  return { socket, lastMessage, connectionStatus, sendMessage };
};

// Risk Dashboard Component
const RiskDashboard: React.FC = () => {
  // State
  const [riskAssessments, setRiskAssessments] = useState<RiskAssessment[]>([]);
  const [statistics, setStatistics] = useState<RiskStatistics | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance | null>(null);
  const [selectedTab, setSelectedTab] = useState(0);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000);
  const [filters, setFilters] = useState({
    riskLevel: 'ALL',
    timeRange: '1h',
    amountRange: [0, 100000],
  });
  const [selectedAssessment, setSelectedAssessment] = useState<RiskAssessment | null>(null);
  const [explanationDialogOpen, setExplanationDialogOpen] = useState(false);
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);
  const [notifications, setNotifications] = useState<Array<{ id: string; message: string; type: 'info' | 'warning' | 'error' | 'success'; timestamp: string }>>([]);

  // WebSocket connection
  const { lastMessage, connectionStatus, sendMessage } = useWebSocket('ws://localhost:8080/ws');

  // API functions
  const fetchRiskAssessments = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/risk-assessments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filters),
      });
      const data = await response.json();
      setRiskAssessments(data);
    } catch (error) {
      console.error('Failed to fetch risk assessments:', error);
    }
  }, [filters]);

  const fetchStatistics = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/statistics');
      const data = await response.json();
      setStatistics(data);
    } catch (error) {
      console.error('Failed to fetch statistics:', error);
    }
  }, []);

  const fetchSystemMetrics = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/metrics');
      const data = await response.json();
      setSystemMetrics(data);
    } catch (error) {
      console.error('Failed to fetch system metrics:', error);
    }
  }, []);

  const fetchModelPerformance = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/model-performance');
      const data = await response.json();
      setModelPerformance(data);
    } catch (error) {
      console.error('Failed to fetch model performance:', error);
    }
  }, []);

  // Effects
  useEffect(() => {
    if (lastMessage) {
      switch (lastMessage.type) {
        case 'risk_assessment':
          setRiskAssessments(prev => [lastMessage.data, ...prev.slice(0, 99)]);
          break;
        case 'statistics_update':
          setStatistics(lastMessage.data);
          break;
        case 'system_metrics':
          setSystemMetrics(lastMessage.data);
          break;
        case 'model_performance':
          setModelPerformance(lastMessage.data);
          break;
        case 'notification':
          setNotifications(prev => [...prev, lastMessage.data]);
          break;
      }
    }
  }, [lastMessage]);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        fetchRiskAssessments();
        fetchStatistics();
        fetchSystemMetrics();
        fetchModelPerformance();
      }, refreshInterval);

      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval, fetchRiskAssessments, fetchStatistics, fetchSystemMetrics, fetchModelPerformance]);

  useEffect(() => {
    fetchRiskAssessments();
    fetchStatistics();
    fetchSystemMetrics();
    fetchModelPerformance();
  }, [fetchRiskAssessments, fetchStatistics, fetchSystemMetrics, fetchModelPerformance]);

  // Computed values
  const riskLevelColors = {
    LOW: '#4caf50',
    MEDIUM: '#ff9800',
    HIGH: '#f44336',
    CRITICAL: '#9c27b0',
  };

  const riskLevelIcons = {
    LOW: <CheckCircle />,
    MEDIUM: <Warning />,
    HIGH: <Error />,
    CRITICAL: <Security />,
  };

  const chartData = useMemo(() => {
    const now = new Date();
    const data = [];
    
    for (let i = 23; i >= 0; i--) {
      const time = subHours(now, i);
      const hourAssessments = riskAssessments.filter(assessment => {
        const assessmentTime = parseISO(assessment.timestamp);
        return assessmentTime >= subHours(now, i + 1) && assessmentTime < subHours(now, i);
      });
      
      data.push({
        time: format(time, 'HH:mm'),
        assessments: hourAssessments.length,
        highRisk: hourAssessments.filter(a => a.riskLevel === 'HIGH' || a.riskLevel === 'CRITICAL').length,
        averageRiskScore: hourAssessments.length > 0 ? 
          hourAssessments.reduce((sum, a) => sum + a.riskScore, 0) / hourAssessments.length : 0,
      });
    }
    
    return data;
  }, [riskAssessments]);

  // Event handlers
  const handleRefresh = () => {
    fetchRiskAssessments();
    fetchStatistics();
    fetchSystemMetrics();
    fetchModelPerformance();
  };

  const handleAssessmentClick = (assessment: RiskAssessment) => {
    setSelectedAssessment(assessment);
    setExplanationDialogOpen(true);
  };

  const handleExportData = () => {
    const dataStr = JSON.stringify(riskAssessments, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `risk-assessments-${format(new Date(), 'yyyy-MM-dd-HH-mm-ss')}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Components
  const ConnectionStatus = () => (
    <Chip
      icon={connectionStatus === 'connected' ? <CloudDone /> : <CloudOff />}
      label={connectionStatus.toUpperCase()}
      color={connectionStatus === 'connected' ? 'success' : 'error'}
      size="small"
    />
  );

  const RiskLevelChip = ({ level }: { level: string }) => (
    <Chip
      icon={riskLevelIcons[level as keyof typeof riskLevelIcons]}
      label={level}
      style={{ backgroundColor: riskLevelColors[level as keyof typeof riskLevelColors], color: 'white' }}
      size="small"
    />
  );

  const SystemHealthCard = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          System Health
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Box display="flex" alignItems="center" mb={1}>
              <Memory fontSize="small" color="primary" />
              <Typography variant="body2" ml={1}>
                CPU: {systemMetrics?.cpuUsage.toFixed(1)}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={systemMetrics?.cpuUsage || 0} 
              color={systemMetrics && systemMetrics.cpuUsage > 80 ? 'error' : 'primary'}
            />
          </Grid>
          <Grid item xs={6}>
            <Box display="flex" alignItems="center" mb={1}>
              <Storage fontSize="small" color="secondary" />
              <Typography variant="body2" ml={1}>
                Memory: {systemMetrics?.memoryUsage.toFixed(1)}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={systemMetrics?.memoryUsage || 0} 
              color={systemMetrics && systemMetrics.memoryUsage > 80 ? 'error' : 'secondary'}
            />
          </Grid>
          <Grid item xs={6}>
            <Box display="flex" alignItems="center" mb={1}>
              <NetworkCheck fontSize="small" color="info" />
              <Typography variant="body2" ml={1}>
                Latency: {systemMetrics?.networkLatency.toFixed(0)}ms
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Box display="flex" alignItems="center" mb={1}>
              <Speed fontSize="small" color="success" />
              <Typography variant="body2" ml={1}>
                RPS: {systemMetrics?.requestsPerSecond.toFixed(0)}
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );

  const StatisticsCard = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Risk Statistics
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="h4" color="primary">
              {statistics?.totalAssessments.toLocaleString()}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Total Assessments
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="h4" color="error">
              {statistics?.highRiskCount.toLocaleString()}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              High Risk
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="h4" color="warning">
              {statistics?.averageRiskScore.toFixed(3)}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Avg Risk Score
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="h4" color="success">
              {statistics?.successRate.toFixed(1)}%
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Success Rate
            </Typography>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );

  const ModelPerformanceCard = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Model Performance
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="h4" color="primary">
              {modelPerformance?.accuracy.toFixed(3)}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Accuracy
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="h4" color="secondary">
              {modelPerformance?.f1Score.toFixed(3)}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              F1 Score
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="h4" color="info">
              {modelPerformance?.auc.toFixed(3)}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              AUC
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="h4" color={modelPerformance && modelPerformance.driftScore > 0.1 ? 'error' : 'success'}>
              {modelPerformance?.driftScore.toFixed(3)}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Drift Score
            </Typography>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );

  const RiskAssessmentsTable = () => (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            Recent Risk Assessments
          </Typography>
          <Box>
            <IconButton onClick={handleRefresh} size="small">
              <Refresh />
            </IconButton>
            <IconButton onClick={handleExportData} size="small">
              <Download />
            </IconButton>
          </Box>
        </Box>
        <TableContainer component={Paper} style={{ maxHeight: 400 }}>
          <Table stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell>Transaction ID</TableCell>
                <TableCell>Customer ID</TableCell>
                <TableCell>Amount</TableCell>
                <TableCell>Risk Score</TableCell>
                <TableCell>Risk Level</TableCell>
                <TableCell>Action</TableCell>
                <TableCell>Time</TableCell>
                <TableCell>Details</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {riskAssessments.slice(0, 50).map((assessment) => (
                <TableRow key={assessment.id} hover>
                  <TableCell>{assessment.transactionId}</TableCell>
                  <TableCell>{assessment.customerId}</TableCell>
                  <TableCell>${assessment.amount.toLocaleString()}</TableCell>
                  <TableCell>
                    <Box display="flex" alignItems="center">
                      <LinearProgress
                        variant="determinate"
                        value={assessment.riskScore * 100}
                        color={assessment.riskLevel === 'HIGH' || assessment.riskLevel === 'CRITICAL' ? 'error' : 'primary'}
                        style={{ width: 60, marginRight: 8 }}
                      />
                      <Typography variant="body2">
                        {assessment.riskScore.toFixed(3)}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <RiskLevelChip level={assessment.riskLevel} />
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={assessment.recommendedAction}
                      color={assessment.recommendedAction === 'APPROVE' ? 'success' : 
                             assessment.recommendedAction === 'REVIEW' ? 'warning' : 'error'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {format(parseISO(assessment.timestamp), 'HH:mm:ss')}
                  </TableCell>
                  <TableCell>
                    <IconButton
                      size="small"
                      onClick={() => handleAssessmentClick(assessment)}
                    >
                      <Info />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );

  const ExplanationDialog = () => (
    <Dialog
      open={explanationDialogOpen}
      onClose={() => setExplanationDialogOpen(false)}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        Risk Assessment Explanation
        <IconButton
          onClick={() => setExplanationDialogOpen(false)}
          style={{ position: 'absolute', right: 8, top: 8 }}
        >
          <Close />
        </IconButton>
      </DialogTitle>
      <DialogContent>
        {selectedAssessment?.explanation && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Feature Importance
            </Typography>
            <List>
              {Object.entries(selectedAssessment.explanation.featureImportance)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 10)
                .map(([feature, importance]) => (
                  <ListItem key={feature}>
                    <ListItemText
                      primary={feature}
                      secondary={`${(importance * 100).toFixed(1)}%`}
                    />
                    <LinearProgress
                      variant="determinate"
                      value={importance * 100}
                      style={{ width: 100 }}
                    />
                  </ListItem>
                ))}
            </List>
            
            <Divider style={{ margin: '16px 0' }} />
            
            <Typography variant="h6" gutterBottom>
              Decision Path
            </Typography>
            <List>
              {selectedAssessment.explanation.decisionPath.map((step, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    <Typography variant="body2" color="primary">
                      {index + 1}.
                    </Typography>
                  </ListItemIcon>
                  <ListItemText primary={step} />
                </ListItem>
              ))}
            </List>
          </Box>
        )}
      </DialogContent>
    </Dialog>
  );

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          Risk Assessment Dashboard
        </Typography>
        <Box display="flex" alignItems="center" gap={2}>
          <ConnectionStatus />
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
            }
            label="Auto Refresh"
          />
          <IconButton onClick={handleRefresh}>
            <Refresh />
          </IconButton>
          <IconButton onClick={() => setSettingsDialogOpen(true)}>
            <Settings />
          </IconButton>
        </Box>
      </Box>

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={selectedTab} onChange={(_, value) => setSelectedTab(value)}>
          <Tab label="Overview" />
          <Tab label="Assessments" />
          <Tab label="Analytics" />
          <Tab label="Model Performance" />
          <Tab label="System Health" />
        </Tabs>
      </Box>

      {/* Tab Content */}
      {selectedTab === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <StatisticsCard />
          </Grid>
          <Grid item xs={12} md={6}>
            <SystemHealthCard />
          </Grid>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Risk Assessment Trends (24 Hours)
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <RechartsTooltip />
                    <Line type="monotone" dataKey="assessments" stroke="#8884d8" strokeWidth={2} />
                    <Line type="monotone" dataKey="highRisk" stroke="#ff7300" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {selectedTab === 1 && (
        <RiskAssessmentsTable />
      )}

      {selectedTab === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Risk Level Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <RechartsPieChart>
                    <Pie
                      data={[
                        { name: 'Low', value: statistics?.lowRiskCount || 0, color: '#4caf50' },
                        { name: 'Medium', value: statistics?.mediumRiskCount || 0, color: '#ff9800' },
                        { name: 'High', value: statistics?.highRiskCount || 0, color: '#f44336' },
                      ]}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                    >
                      {data.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                  </RechartsPieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Processing Time Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <RechartsBarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <RechartsTooltip />
                    <Bar dataKey="averageRiskScore" fill="#8884d8" />
                  </RechartsBarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {selectedTab === 3 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <ModelPerformanceCard />
          </Grid>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Model Performance Over Time
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <RechartsTooltip />
                    <Line type="monotone" dataKey="averageRiskScore" stroke="#8884d8" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {selectedTab === 4 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <SystemHealthCard />
          </Grid>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  System Metrics
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={3}>
                    <Box textAlign="center">
                      <Typography variant="h4" color="primary">
                        {systemMetrics?.requestsPerSecond.toFixed(0)}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Requests/sec
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={3}>
                    <Box textAlign="center">
                      <Typography variant="h4" color="secondary">
                        {systemMetrics?.activeConnections}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Active Connections
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={3}>
                    <Box textAlign="center">
                      <Typography variant="h4" color="info">
                        {systemMetrics?.cacheHitRate.toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Cache Hit Rate
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={3}>
                    <Box textAlign="center">
                      <Typography variant="h4" color="error">
                        {systemMetrics?.errorRate.toFixed(2)}%
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Error Rate
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Dialogs */}
      <ExplanationDialog />

      {/* Notifications */}
      <Snackbar
        open={notifications.length > 0}
        autoHideDuration={6000}
        onClose={() => setNotifications([])}
      >
        <Alert severity="info">
          {notifications[notifications.length - 1]?.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default RiskDashboard;
