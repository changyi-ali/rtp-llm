package org.flexlb.engine.grpc.client;

import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.EngineAddressResolver;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class EngineGrpcClient extends org.flexlb.engine.grpc.EngineGrpcClient {
    public EngineGrpcClient(EngineAddressResolver resolver, GrpcChannelFactory factory,
            GrpcReporter reporter, CacheMatchConfiguration configuration) {
        this(resolver, factory, reporter, configuration, 5000L);
    }

    @Autowired
    public EngineGrpcClient(EngineAddressResolver resolver, GrpcChannelFactory factory,
            GrpcReporter reporter, CacheMatchConfiguration configuration,
            @Value("${flexlb.engine-grpc.enqueue-timeout-ms:5000}") long enqueueTimeoutMillis) {
        super(resolver, factory.getExecutor(), factory.getEventLoopGroup(), reporter,
                factory.getConnectTimeoutMillis(), enqueueTimeoutMillis);
        cacheStatusEnabled = !configuration.isKvcmEnabled();
    }
}
