    def calculate_volume(self):
        """Estimate the volume if not provided (using the ideal gas law)."""
        if self.volume is None:
            return (self.r * self.temperature) / (self.pressure * 1e5)  # Convert pressure to Pa
        return self.volume
    
