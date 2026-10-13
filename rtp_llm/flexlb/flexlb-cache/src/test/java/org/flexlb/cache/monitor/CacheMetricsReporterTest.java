package org.flexlb.cache.monitor;
import org.flexlb.cache.telemetry.CacheMetricsReporter;

import org.flexlb.cache.core.RecentCacheKeyWindow;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;

import static org.flexlb.constant.MetricConstant.CACHE_AFFINITY_DECISION;
import static org.flexlb.constant.MetricConstant.CACHE_CLUSTER_RECENT_HISTORY_COVERAGE_MS;
import static org.flexlb.constant.MetricConstant.CACHE_CLUSTER_RECENT_HISTORY_READY;
import static org.flexlb.constant.MetricConstant.CACHE_CLUSTER_RECENT_HIT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_CLUSTER_RECENT_INPUT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_ENGINE_LOCAL_BYTES;
import static org.flexlb.constant.MetricConstant.CACHE_ENGINE_LOCAL_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_RECENT_KEY_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_RECENT_KEY_TOTAL_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_ROUTING_CANDIDATE_MAX_HIT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_ROUTING_SELECTED_MATCH_TOTAL_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_THEORY_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_THEORY_HIT_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_THEORY_TOTAL_COUNT;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class CacheMetricsReporterTest {

    @Mock private FlexMonitor monitor;

    private CacheMetricsReporter reporter;

    @BeforeEach
    void setUp() {
        reporter = new CacheMetricsReporter();
        try {
            Field monitorField = CacheMetricsReporter.class.getDeclaredField("monitor");
            monitorField.setAccessible(true);
            monitorField.set(reporter, monitor);
        } catch (Exception e) {
            fail("Failed to inject monitor: " + e.getMessage());
        }
    }

    @Test
    void should_not_publish_a_cluster_ratio_from_partial_history() {
        RecentCacheKeyWindow.Snapshot snapshot = mock(RecentCacheKeyWindow.Snapshot.class);
        when(snapshot.getTimeWindowMs()).thenReturn(1800000L);
        when(snapshot.getHistoryCoverageMs()).thenReturn(1000L);
        reporter.reportClusterRecentCacheKeyHitMetrics(snapshot, 0L, 216513L);
        FlexMetricTags tags = FlexMetricTags.of("timeWindowMs", "1800000", "isMaster", "true");
        verify(monitor).report(CACHE_CLUSTER_RECENT_HISTORY_COVERAGE_MS, tags, 1000L);
        verify(monitor).report(CACHE_CLUSTER_RECENT_HISTORY_READY, tags, 0L);
        verifyNoMoreInteractions(monitor);
    }

    @Test
    void should_report_additive_cluster_token_counts_including_cold_requests() {
        RecentCacheKeyWindow.Snapshot snapshot = mock(RecentCacheKeyWindow.Snapshot.class);
        when(snapshot.getTimeWindowMs()).thenReturn(1800000L);
        when(snapshot.getHistoryCoverageMs()).thenReturn(1800000L);
        when(snapshot.isHistoryComplete()).thenReturn(true);
        reporter.reportClusterRecentCacheKeyHitMetrics(snapshot, 0L, 300L);
        reporter.reportClusterRecentCacheKeyHitMetrics(snapshot, 216064L, 216513L);
        FlexMetricTags tags = FlexMetricTags.of("timeWindowMs", "1800000", "isMaster", "true");
        verify(monitor).report(CACHE_CLUSTER_RECENT_HIT_TOKENS, tags, 0L);
        verify(monitor).report(CACHE_CLUSTER_RECENT_INPUT_TOKENS, tags, 300L);
        verify(monitor).report(CACHE_CLUSTER_RECENT_HIT_TOKENS, tags, 216064L);
        verify(monitor).report(CACHE_CLUSTER_RECENT_INPUT_TOKENS, tags, 216513L);
    }


}
