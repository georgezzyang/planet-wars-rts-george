# ---------- Stage 1: Build with Gradle wrapper ----------
FROM eclipse-temurin:21-jdk AS builder

WORKDIR /home/build

# Copy the entire project (gradle wrapper, sources, build files)
COPY . .

# Make wrapper executable (Windows clones may strip the +x bit) and build the shadow jar.
# We build only :app:shadowJar (skips tests) for a fast self-contained build.
RUN chmod +x gradlew && ./gradlew :app:shadowJar --no-daemon -x test

# ---------- Stage 2: Slim runtime ----------
FROM eclipse-temurin:21-jre

WORKDIR /app

COPY --from=builder /home/build/app/build/libs/client-server.jar app.jar

# Submission system expects WebSocket server on port 8080
EXPOSE 8080

CMD ["java", "-jar", "app.jar"]
