pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

// Lets `jvmToolchain(17)` download a JDK 17 when none is installed, so contributors need only a
// JVM capable of running Gradle itself.
plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "0.8.0"
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

rootProject.name = "presence"

include(":shared")
include(":harness")
